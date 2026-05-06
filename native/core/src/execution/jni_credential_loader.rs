// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! JNI-backed AWS credential loader for Comet's native Iceberg reader.
//!
//! Implements `reqsign::AwsCredentialLoad` by calling back to a Java
//! `CometCredentialProvider.resolveCredentials(ResolveContext)` via JNI. OpenDAL calls
//! `load_credential()` on every S3 request (before signing). This implementation
//! caches credentials and only performs the JNI callback when credentials are
//! near expiry.
//!
//! With the catalog-scoped provider design (see `CatalogScopedCredentialProvider` on the JVM
//! side), the `credential_provider` reference points to a JVM-singleton object shared across
//! all tasks on the executor JVM. This loader is constructed PER Iceberg scan and binds the
//! provider to a specific `table_location` — every JNI callback wraps that location in a
//! `ResolveContext` so the single shared provider can dispatch internally (e.g. lookup the
//! right per-table delegate). `ResolveContext` is the SPI-stable extension point: future
//! per-call fields (snapshot id, operation type, etc.) will be added to that class without
//! changing the JNI signature shape.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use iceberg_storage_opendal::{AwsCredential, AwsCredentialLoad};
use jni::objects::{Global, JClass, JMethodID, JObject, JObjectArray, JString, JValue};
use jni::signature::ReturnType;

use crate::execution::operators::ExecutionError;
use crate::jvm_bridge::{check_exception, JVMClasses};

/// Initial capacity hint for the local reference frame used on each credential resolution
/// call. A single call materializes a small, fixed set of JNI locals (context object, result
/// array, 4 element strings); the JVM grows the frame transparently if the hint is too low.
const CALL_LOCAL_FRAME_CAPACITY: usize = 16;

pub struct JniAwsCredentialLoader {
    credential_provider: Arc<Global<JObject<'static>>>,
    /// Method id for `String[] resolveCredentials(ResolveContext)`.
    resolve_method_id: JMethodID,
    /// Method id for the single-arg `ResolveContext(String tableLocation)` ctor. The 2-arg
    /// ctor (with a properties map) is intentionally not used from the native side: per-call
    /// properties are reserved for future evolution and have no source to populate them from
    /// in Rust today.
    resolve_context_ctor: JMethodID,
    /// Global ref to `ResolveContext.class`, cached at construction time so we don't pay
    /// `FindClass` on the hot path and don't rely on whichever class loader is active when
    /// the first S3 request fires (historically a source of Spark classloader bugs).
    resolve_context_class: Global<JClass<'static>>,
    /// Global ref to the Java `String` for this loader's table location. Bound once at
    /// construction so every JNI call reuses the same Java string instance.
    table_location_jobj: Global<JString<'static>>,
    /// Table location string, retained only for log/error context.
    table_location: String,
    cached: RwLock<Option<CachedCredential>>,
    min_ttl: Duration,
}

struct CachedCredential {
    credential: AwsCredential,
    expires_at_ms: i64,
    fetched_at: std::time::Instant,
}

impl JniAwsCredentialLoader {
    pub fn new(
        credential_provider: Arc<Global<JObject<'static>>>,
        table_location: String,
        min_ttl: Duration,
    ) -> Result<Self, ExecutionError> {
        log::debug!(
            "Constructing JniAwsCredentialLoader table_location={table_location} \
             min_ttl={min_ttl:?}"
        );
        let (resolve_method_id, resolve_context_ctor, resolve_context_class, table_location_jobj) =
            JVMClasses::with_env(|env| {
                let resolve_method_id = env
                    .get_method_id(
                        jni::jni_str!("org/apache/comet/iceberg/CometCredentialProvider"),
                        jni::jni_str!("resolveCredentials"),
                        jni::jni_sig!(
                            "(Lorg/apache/comet/iceberg/ResolveContext;)[Ljava/lang/String;"
                        ),
                    )
                    .map_err(|e| {
                        ExecutionError::GeneralError(format!(
                            "Failed to get JNI method ID for \
                             CometCredentialProvider.resolveCredentials(ResolveContext). \
                             Is the provider class on the executor classpath and implementing \
                             the current SPI? {}",
                            e
                        ))
                    })?;

                let context_class_local = env
                    .find_class(jni::jni_str!("org/apache/comet/iceberg/ResolveContext"))
                    .map_err(|e| {
                        ExecutionError::GeneralError(format!(
                            "Failed to find class ResolveContext: {}",
                            e
                        ))
                    })?;

                let ctor_id = env
                    .get_method_id(
                        &context_class_local,
                        jni::jni_str!("<init>"),
                        jni::jni_sig!("(Ljava/lang/String;)V"),
                    )
                    .map_err(|e| {
                        ExecutionError::GeneralError(format!(
                            "Failed to get JNI ctor ID for ResolveContext(String): {}",
                            e
                        ))
                    })?;

                let class_global = env.new_global_ref(context_class_local).map_err(|e| {
                    ExecutionError::GeneralError(format!(
                        "Failed to create global ref for ResolveContext class: {}",
                        e
                    ))
                })?;

                let table_loc_local = env.new_string(&table_location).map_err(|e| {
                    ExecutionError::GeneralError(format!(
                        "Failed to allocate Java String for tableLocation: {}",
                        e
                    ))
                })?;
                let table_loc_global = env.new_global_ref(table_loc_local).map_err(|e| {
                    ExecutionError::GeneralError(format!(
                        "Failed to create global ref for tableLocation String: {}",
                        e
                    ))
                })?;

                Ok::<_, ExecutionError>((
                    resolve_method_id,
                    ctor_id,
                    class_global,
                    table_loc_global,
                ))
            })?;

        Ok(Self {
            credential_provider,
            resolve_method_id,
            resolve_context_ctor,
            resolve_context_class,
            table_location_jobj,
            table_location,
            cached: RwLock::new(None),
            min_ttl,
        })
    }

    fn is_valid(cached: &CachedCredential, min_ttl: &Duration) -> bool {
        if cached.expires_at_ms < 0 {
            // No expiry info — treat as valid for min_ttl from fetch time
            return cached.fetched_at.elapsed() < *min_ttl;
        }
        let now_ms = chrono::Utc::now().timestamp_millis();
        let buffer_ms = min_ttl.as_millis() as i64;
        cached.expires_at_ms > now_ms + buffer_ms
    }

    fn call_jvm(&self) -> Result<AwsCredential, ExecutionError> {
        JVMClasses::with_env(|env| {
            // Run the entire per-call JNI dance inside a local reference frame so every
            // intermediate local (ResolveContext, the result array, and each String element)
            // is freed when we return. Without this, a long-running Iceberg scan would
            // accumulate local refs per S3 request and eventually blow past the JVM's
            // local-ref table limit.
            //
            // `with_local_frame` requires `E: From<jni::errors::Error>`, so we nest a
            // `Result<Result<_, ExecutionError>, jni::errors::Error>` and flatten on the way
            // out — the inner `ExecutionError` payload carries the real call-site failure,
            // while the outer `jni::errors::Error` is reserved for frame push/pop issues.
            let nested: Result<Result<AwsCredential, ExecutionError>, jni::errors::Error> = env
                .with_local_frame(CALL_LOCAL_FRAME_CAPACITY, |env| {
                    Ok(self.call_jvm_in_frame(env))
                });
            match nested {
                Ok(inner) => inner,
                Err(e) => Err(ExecutionError::GeneralError(format!(
                    "Failed to push JNI local reference frame for table_location={}: {}",
                    self.table_location, e
                ))),
            }
        })
    }

    /// Body of [`call_jvm`] executed inside a fresh local-reference frame.
    fn call_jvm_in_frame(&self, env: &mut jni::Env<'_>) -> Result<AwsCredential, ExecutionError> {
        let instance = self.credential_provider.as_obj();
        let table_loc_jstr = self.table_location_jobj.as_obj();

        let context_obj = unsafe {
            env.new_object_unchecked(
                &self.resolve_context_class,
                self.resolve_context_ctor,
                &[JValue::Object(table_loc_jstr).as_jni()],
            )
        }
        .map_err(|e| {
            ExecutionError::GeneralError(format!(
                "Failed to construct ResolveContext for table_location={}: {}",
                self.table_location, e
            ))
        })?;

        let result = unsafe {
            env.call_method_unchecked(
                instance,
                self.resolve_method_id,
                ReturnType::Array,
                &[JValue::Object(&context_obj).as_jni()],
            )
        };

        if let Some(exception) = check_exception(env).map_err(|e| {
            ExecutionError::GeneralError(format!("Failed to check for Java exception: {}", e))
        })? {
            return Err(ExecutionError::GeneralError(format!(
                "Java exception during credential resolution for table_location={}: {}",
                self.table_location, exception
            )));
        }

        let result = result.map_err(|e| {
            ExecutionError::GeneralError(format!(
                "JNI method call failed for resolveCredentials({}): {}",
                self.table_location, e
            ))
        })?;

        let result_obj = result.l().map_err(|e| {
            ExecutionError::GeneralError(format!("Failed to extract result: {}", e))
        })?;

        let array = unsafe { JObjectArray::<JObject>::from_raw(env, result_obj.into_raw()) };

        let len = array.len(env).map_err(|e| {
            ExecutionError::GeneralError(format!("Failed to get array length: {}", e))
        })?;

        if len < 4 {
            return Err(ExecutionError::GeneralError(format!(
                "resolveCredentials({}) returned {} elements, expected 4",
                self.table_location, len
            )));
        }

        let mut get_string = |idx: usize| -> Result<String, ExecutionError> {
            let obj: JObject = array.get_element(env, idx).map_err(|e| {
                ExecutionError::GeneralError(format!("Failed to get array element {}: {}", idx, e))
            })?;
            let jstr = unsafe { JString::from_raw(env, obj.into_raw()) };
            jstr.try_to_string(env).map_err(|e| {
                ExecutionError::GeneralError(format!(
                    "Failed to convert string at index {}: {}",
                    idx, e
                ))
            })
        };

        let access_key_id = get_string(0)?;
        let secret_access_key = get_string(1)?;
        let session_token = get_string(2)?;
        let expires_at_str = get_string(3)?;

        let expires_at_ms: i64 = expires_at_str.parse().unwrap_or(-1);

        let session_token_opt = if session_token.is_empty() {
            None
        } else {
            Some(session_token)
        };

        let expires_in = if expires_at_ms > 0 {
            chrono::DateTime::from_timestamp_millis(expires_at_ms)
        } else {
            None
        };

        let credential = AwsCredential {
            access_key_id,
            secret_access_key,
            session_token: session_token_opt,
            expires_in,
        };

        log::trace!(
            "JVM returned credentials table_location={} expires_at_ms={} \
             session_token_present={}",
            self.table_location,
            expires_at_ms,
            credential.session_token.is_some()
        );

        Ok(credential)
    }
}

// Safety: JNI global refs are safe to share across threads.
// JVMClasses::with_env() handles thread attachment.
unsafe impl Send for JniAwsCredentialLoader {}
unsafe impl Sync for JniAwsCredentialLoader {}

#[async_trait]
impl AwsCredentialLoad for JniAwsCredentialLoader {
    async fn load_credential(
        &self,
        _client: reqwest::Client,
    ) -> anyhow::Result<Option<AwsCredential>> {
        // Fast path: check cache with read lock
        {
            let guard = self
                .cached
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
            if let Some(ref cached) = *guard {
                if Self::is_valid(cached, &self.min_ttl) {
                    return Ok(Some(cached.credential.clone()));
                }
            }
        }

        // Slow path: acquire write lock, re-check (another thread may have refreshed),
        // then do JNI callback only if still needed.
        let mut guard = self
            .cached
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;
        if let Some(ref cached) = *guard {
            if Self::is_valid(cached, &self.min_ttl) {
                return Ok(Some(cached.credential.clone()));
            }
        }

        log::debug!(
            "Cache miss/expired -- calling JVM resolveCredentials(table_location={})",
            self.table_location
        );
        let credential = self
            .call_jvm()
            .map_err(|e| anyhow::anyhow!("JNI credential refresh failed: {}", e))?;

        let expires_at_ms = credential
            .expires_in
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(-1);

        *guard = Some(CachedCredential {
            credential: credential.clone(),
            expires_at_ms,
            fetched_at: std::time::Instant::now(),
        });

        Ok(Some(credential))
    }
}

impl std::fmt::Debug for JniAwsCredentialLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JniAwsCredentialLoader")
            .field("table_location", &self.table_location)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_credential(expires_in: Option<chrono::DateTime<chrono::Utc>>) -> CachedCredential {
        CachedCredential {
            credential: AwsCredential {
                access_key_id: "AKIA_TEST".to_string(),
                secret_access_key: "secret".to_string(),
                session_token: Some("token".to_string()),
                expires_in,
            },
            expires_at_ms: expires_in.map(|dt| dt.timestamp_millis()).unwrap_or(-1),
            fetched_at: std::time::Instant::now(),
        }
    }

    #[test]
    fn test_is_valid_with_future_expiry() {
        let future = chrono::Utc::now() + chrono::TimeDelta::try_hours(1).unwrap();
        let cached = make_credential(Some(future));
        let ttl = Duration::from_secs(300);
        assert!(JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }

    #[test]
    fn test_is_valid_with_near_expiry() {
        let soon = chrono::Utc::now() + chrono::TimeDelta::try_seconds(60).unwrap();
        let cached = make_credential(Some(soon));
        let ttl = Duration::from_secs(300); // 5 min buffer
        assert!(!JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }

    #[test]
    fn test_is_valid_with_past_expiry() {
        let past = chrono::Utc::now() - chrono::TimeDelta::try_hours(1).unwrap();
        let cached = make_credential(Some(past));
        let ttl = Duration::from_secs(300);
        assert!(!JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }

    #[test]
    fn test_is_valid_no_expiry_within_ttl() {
        let cached = make_credential(None);
        let ttl = Duration::from_secs(300);
        assert!(JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }

    #[test]
    fn test_is_valid_no_expiry_beyond_ttl() {
        let mut cached = make_credential(None);
        cached.fetched_at = std::time::Instant::now() - Duration::from_secs(600);
        let ttl = Duration::from_secs(300);
        assert!(!JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }

    #[test]
    fn test_is_valid_treats_negative_expires_at_as_no_expiry() {
        // When the JVM signals "no expiry" via expires_at_ms=-1, the loader should fall back
        // to the fetched_at + min_ttl liveness check rather than returning false outright.
        let mut cached = make_credential(None);
        cached.expires_at_ms = -1;
        cached.fetched_at = std::time::Instant::now();
        let ttl = Duration::from_secs(300);
        assert!(JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }

    #[test]
    fn test_is_valid_zero_min_ttl_with_future_expiry() {
        // A zero buffer means "valid as long as not yet expired"; future expiries pass.
        let future = chrono::Utc::now() + chrono::TimeDelta::try_seconds(1).unwrap();
        let cached = make_credential(Some(future));
        let ttl = Duration::from_secs(0);
        assert!(JniAwsCredentialLoader::is_valid(&cached, &ttl));
    }
}
