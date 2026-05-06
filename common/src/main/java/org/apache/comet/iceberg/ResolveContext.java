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

package org.apache.comet.iceberg;

import java.io.Serializable;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Per-call context passed to {@link CometCredentialProvider#resolveCredentials(ResolveContext)}.
 */
public final class ResolveContext implements Serializable {

  private static final long serialVersionUID = 1L;

  private final String tableLocation;
  private final Map<String, String> properties;

  public ResolveContext(String tableLocation, Map<String, String> properties) {
    this.tableLocation = tableLocation;
    this.properties =
        properties == null ? Collections.emptyMap() : Collections.unmodifiableMap(properties);
  }

  public ResolveContext(String tableLocation) {
    this(tableLocation, Collections.emptyMap());
  }

  public String tableLocation() {
    return tableLocation == null ? "" : tableLocation;
  }

  public Map<String, String> properties() {
    return properties;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof ResolveContext)) return false;
    ResolveContext that = (ResolveContext) o;
    return Objects.equals(tableLocation, that.tableLocation)
        && Objects.equals(properties, that.properties);
  }

  @Override
  public int hashCode() {
    return Objects.hash(tableLocation, properties);
  }

  @Override
  public String toString() {
    return "ResolveContext{tableLocation='" + tableLocation + "', properties=" + properties + '}';
  }
}
