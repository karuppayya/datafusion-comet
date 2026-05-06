/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.comet

import java.util.concurrent.ConcurrentHashMap

import scala.jdk.CollectionConverters._

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.{SparkListener, SparkListenerApplicationEnd}

import org.apache.comet.iceberg.CometCredentialProvider

/**
 * Driver-side broadcast cache for [[CometCredentialProvider]] instances.
 */
private[spark] object CatalogScopedCredentialProvider extends Logging {

  /** Cache keyed by (providerClass, catalogIdentity). */
  private val cache =
    new ConcurrentHashMap[CatalogKey, Broadcast[CometCredentialProvider]]()

  @volatile private var listenerRegistered = false

  /**
   * Get (or build) the broadcasted credential provider for the given catalog.
   */
  def getOrCreate(
      sc: SparkContext,
      className: String,
      catalogProperties: Map[String, String]): Broadcast[CometCredentialProvider] = {

    ensureCleanupListener(sc)

    val identity = fingerprint(catalogProperties)
    val key = CatalogKey(className, identity)

    val existing = cache.get(key)
    if (existing != null) {
      logDebug(
        s"Reusing broadcast credential provider id=${existing.id} " +
          s"className=$className catalogIdentity=$identity")
      return existing
    }

    cache.computeIfAbsent(
      key,
      _ => {
        val startMs = System.currentTimeMillis()
        val provider = instantiateAndInit(className, catalogProperties)
        val initElapsedMs = System.currentTimeMillis() - startMs

        val broadcast = sc.broadcast(provider)
        logInfo(
          s"Broadcast credential provider created id=${broadcast.id} " +
            s"className=$className catalogIdentity=$identity " +
            s"initElapsedMs=$initElapsedMs")
        broadcast
      })
  }

  private def fingerprint(props: Map[String, String]): Int =
    props.toSeq.sorted.hashCode()

  private def instantiateAndInit(
      className: String,
      catalogProperties: Map[String, String]): CometCredentialProvider = {
    // Use the context classloader so classes from the application JAR (loaded by Spark's user
    // classloader) are visible, not just classes on Comet's own extraClassPath.
    val classLoader = Thread.currentThread().getContextClassLoader
    // scalastyle:off classforname
    val clazz = Class.forName(className, true, classLoader)
    // scalastyle:on classforname
    val provider = clazz
      .getDeclaredConstructor()
      .newInstance() // scalastyle:ignore
      .asInstanceOf[CometCredentialProvider]
    provider.initialize(catalogProperties.asJava)
    provider
  }

  private def ensureCleanupListener(sc: SparkContext): Unit = {
    if (!listenerRegistered) synchronized {
      if (!listenerRegistered) {
        sc.addSparkListener(new SparkListener {
          override def onApplicationEnd(end: SparkListenerApplicationEnd): Unit = {
            val cachedCount = cache.size
            if (cachedCount > 0) {
              logInfo(s"Unpersisting $cachedCount cached credential provider broadcasts")
            }
            cache.values().forEach { b =>
              try b.unpersist(blocking = false)
              catch {
                case t: Throwable =>
                  logWarning(s"Failed to unpersist broadcast id=${b.id}", t)
              }
            }
            cache.clear()
            // Allow re-registration if a new SparkContext is created in the same JVM
            CatalogScopedCredentialProvider.synchronized {
              listenerRegistered = false
            }
          }
        })
        listenerRegistered = true
      }
    }
  }

  private[comet] def clearForTesting(): Unit = {
    cache.clear()
    synchronized { listenerRegistered = false }
  }

  private[comet] def cacheSizeForTesting(): Int = cache.size

  private case class CatalogKey(className: String, catalogIdentity: Int)
}
