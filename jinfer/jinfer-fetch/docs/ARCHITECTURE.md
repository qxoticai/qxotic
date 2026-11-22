# jinfer-fetch — class architecture

Package `com.qxotic.jinfer.fetch` (5 classes, ~620 lines; JPMS module `com.qxotic.jinfer.fetch`). It is the
**HTTP GET layer** of the stack: "one HTTP GET, two ways to make it" — the JDK `HttpClient` (`JdkTransport`) or
`curl` run as a process (`CurlTransport`). The module has **no project dependencies** (only `java.net.http`) and
compiles with `--release 11`, below jinfer's Java 25, because `toknroll-hf` and `toknroll-gguf` download through
it as well as `jinfer-hub`. A native image excludes `JdkTransport` and so carries no TLS stack: curl and the
system's TLS do the work.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` nested/composed; classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contract, 2b internals
3. [Transport selection and SPI registration](#3-transport-selection-and-spi-registration)
4. [JdkTransport.get](#4-jdktransportget)
5. [CurlTransport.get](#5-curltransportget)
6. [follow: redirects and credential stripping](#6-follow-redirects-and-credential-stripping)
7. [deadline: bounding the body](#7-deadline-bounding-the-body)
8. [Proxies: environment proxy selection](#8-proxies-environment-proxy-selection)
9. [Hardening and native-image notes](#9-hardening-and-native-image-notes)

| Class | Visibility | Role |
|---|---|---|
| `Transport` | public interface | The contract: `get` (one GET, 3xx returned as-is), default `follow` (≤5 redirects, drops `Authorization` off-origin), static `deadline`, `system()`, `offline()`; nested `Response` |
| `Transport.Response` | public nested interface | `Closeable`: `status()`, `headers(name)` (case-insensitive, ordered), default `header(name)` (first value or null), `body()` |
| `JdkTransport` | public final | Service provider: one static `HttpClient` (HTTP/1.1, no redirects, `Proxies.selector()`, 30 s connect timeout); rewrites `ConnectException` to name the host |
| `CurlTransport` | package-private final | `curl -q -sS -K - -D - --suppress-connect-headers -- URL` as a process; parses status + headers off stdout, body checks curl's exit at EOF |
| `Transports` | package-private final | Process-wide singletons made at class load: `SYSTEM` (chosen transport), `OFFLINE`, `DEADLINES` executor; validators `requireHttp`, `requireClean` |
| `Proxies` | package-private final | `HTTP(S)_PROXY` / `NO_PROXY` read as curl reads them → a `ProxySelector`; nested `EnvProxySelector` |

Resources: `META-INF/services/com.qxotic.jinfer.fetch.Transport` → `JdkTransport` (also declared in `module-info`
with `provides … with JdkTransport` and `uses Transport`); `META-INF/native-image/com.qxotic/jinfer-fetch/native-image.properties`
→ `--initialize-at-run-time=com.qxotic.jinfer.fetch`.

---

## 1. Position in the stack

```mermaid
flowchart TB
    subgraph users["Callers"]
        Hub["jinfer-hub: Fetch, RateLimit<br/>model downloads"]
        Hf["toknroll-hf: RepositoryArtifactCache,<br/>HuggingFaceTokenizerLoader"]
        Gg["toknroll-gguf: GGUFMetadataCache"]
    end
    subgraph fetch["jinfer-fetch"]
        T["Transport"]
        Ts["Transports"]
        J["JdkTransport"]
        C["CurlTransport"]
        P["Proxies"]
    end
    Http["java.net.http.HttpClient"]
    Curl["curl process on PATH"]
    Native["jinfer-native image build<br/>excludes JdkTransport provider"]

    Hub -->|"Transport.system().get"| T
    Hf -->|"follow + deadline, system or offline"| T
    Gg -->|"follow + deadline"| T
    T --> Ts
    Ts -->|ServiceLoader| J
    Ts -->|fallback| C
    J --> Http
    J --> P
    C --> Curl
    Native -.->|"-H:ServiceLoaderFeatureExcludeServiceProviders"| J
```

- `jinfer-hub`'s `Fetch` calls `Transport.system().get(...)` and follows redirects itself (`sendOnce`), adding
  rate-limit handling (`RateLimit.requested(Transport.Response, …)` reads `retry-after` / `ratelimit` headers).
- `toknroll-hf` and `toknroll-gguf` hold a `Transport` field (injected; `Transport.system()` by default,
  `Transport.offline()` when `HuggingFaceTokenizerLoader` is cache-only) and call
  `Transport.deadline(transport.follow(uri, headers, timeout), timeout)`.

---

## 2. Class diagrams

### 2a. Contract

```mermaid
classDiagram
    direction LR
    class Closeable { <<interface, external>> }
    class Transport {
        <<interface>>
        +get(URI uri, Map headers, Duration timeout) Response
        +follow(URI uri, Map headers, Duration timeout) Response
        +deadline(Response response, Duration timeout)$ Response
        +system()$ Transport
        +offline()$ Transport
    }
    class Response {
        <<interface>>
        +status() int
        +headers(String name) List~String~
        +header(String name) String
        +body() InputStream
    }
    class JdkTransport {
        <<final>>
        -CLIENT$ HttpClient
        +get(URI, Map, Duration) Response
    }
    class CurlTransport {
        <<final>>
        ~available()$ boolean
        +get(URI, Map, Duration) Response
        -failure(Process, URI)$ IOException
        -line(InputStream)$ String
    }
    Response --|> Closeable
    Transport *-- Response : nested
    JdkTransport ..|> Transport
    CurlTransport ..|> Transport
    JdkTransport ..> Response : anonymous impl
    CurlTransport ..> Response : anonymous impl
```

`get` is the only abstract method, so `Transport` is usable as a lambda: `Transports.OFFLINE` and the
"no HTTP transport" fallback are lambdas. `follow` and `deadline` are written once, against `get` and `Response`,
and serve both transports.

### 2b. Internals

```mermaid
classDiagram
    direction LR
    class Transports {
        <<final>>
        ~SYSTEM$ Transport
        ~OFFLINE$ Transport
        ~DEADLINES$ ScheduledThreadPoolExecutor
        -choose()$ Transport
        -deadlines()$ ScheduledThreadPoolExecutor
        ~requireHttp(URI)$ void
        ~requireClean(Map)$ void
    }
    class Proxies {
        <<final>>
        ~selector()$ ProxySelector
        ~selector(UnaryOperator getenv)$ ProxySelector
        -env(UnaryOperator, String, String)$ String
    }
    class EnvProxySelector {
        <<final>>
        -http Proxy
        -https Proxy
        -noProxy List~String~
        +select(URI) List~Proxy~
        +connectFailed(URI, SocketAddress, IOException) void
    }
    class ProxySelector { <<abstract, external>> }
    class HttpClient { <<external>> }
    class ServiceLoader { <<external>> }
    class Process { <<external>> }
    class Transport { <<interface>> }
    class JdkTransport
    class CurlTransport

    Proxies *-- EnvProxySelector : nested
    EnvProxySelector --|> ProxySelector
    Proxies ..> EnvProxySelector : creates
    JdkTransport --> HttpClient : CLIENT
    JdkTransport ..> Proxies : selector()
    JdkTransport ..> Transports : requireHttp, requireClean
    CurlTransport ..> Transports : requireHttp, requireClean
    CurlTransport ..> Process : ProcessBuilder
    Transports ..> ServiceLoader : load Transport
    Transports ..> CurlTransport : available, new
    Transports --> Transport : SYSTEM, OFFLINE
    Transport ..> Transports : system, offline, DEADLINES
```

---

## 3. Transport selection and SPI registration

`Transports` is initialised once (first use of `Transport.system()`/`offline()`/`deadline()`); its static
fields fix the choice for the process.

```mermaid
flowchart TD
    A["Transport.system()"] --> B["Transports.SYSTEM = choose()"]
    B --> C{"ServiceLoader.load(Transport) has a provider?"}
    C -- "yes: JVM, META-INF/services or module provides" --> D["first provider: JdkTransport"]
    C -- "no: native image excluded it" --> E{"CurlTransport.available()?<br/>curl --version exits 0 within 5 s"}
    E -- yes --> F["new CurlTransport()"]
    E -- no --> G["lambda that throws IOException:<br/>no HTTP transport, install curl"]
```

- Registration is double: `module-info.java` (`provides com.qxotic.jinfer.fetch.Transport with JdkTransport`) for
  the module path, and `META-INF/services/com.qxotic.jinfer.fetch.Transport` for the class path. `JdkTransport` is
  therefore `public` with an implicit no-arg constructor; `CurlTransport` is not a provider and stays package-private.
- `jinfer-native/pom.xml` passes `-H:ServiceLoaderFeatureExcludeServiceProviders=com.qxotic.jinfer.fetch.JdkTransport`,
  so in the image the loader finds nothing and curl is chosen.
- `Transports.OFFLINE` throws `NoSuchFileException("offline: no request to " + uri)` before connecting, so a cache
  miss reads as a missing file.
- `Transports.DEADLINES`: a one-thread `ScheduledThreadPoolExecutor`, daemon thread `fetch-deadline`,
  `setRemoveOnCancelPolicy(true)`.

---

## 4. JdkTransport.get

```mermaid
sequenceDiagram
    participant Caller
    participant J as JdkTransport
    participant Ts as Transports
    participant HC as HttpClient CLIENT (external)
    Caller->>J: get(uri, headers, timeout)
    J->>Ts: requireHttp(uri)
    J->>Ts: requireClean(headers)
    J->>J: HttpRequest.newBuilder(uri).timeout(timeout).GET(), add headers
    Note over J: IllegalArgumentException on a reserved header becomes IOException
    J->>HC: send(request, BodyHandlers.ofInputStream())
    alt ConnectException
        J-->>Caller: new ConnectException naming host:port, cannot find host if UnresolvedAddressException in causes
    else InterruptedException
        J-->>Caller: IOException, interrupt flag restored
    else response
        HC-->>J: HttpResponse of InputStream
        J-->>Caller: Response wrapping statusCode, headers().allValues, body
    end
```

`CLIENT` is built with `HttpClient.Version.HTTP_1_1` (HTTP/2 would multiplex parallel ranged GETs onto one
connection), `Redirect.NEVER`, `proxy(Proxies.selector())` and a 30 s connect timeout. The request `timeout`
bounds the wait for status and headers; the body is bounded by the caller (see §7).

---

## 5. CurlTransport.get

```mermaid
sequenceDiagram
    participant Caller
    participant C as CurlTransport
    participant Ts as Transports
    participant P as curl process (external)
    Caller->>C: get(url, headers, timeout)
    C->>Ts: requireHttp(url), requireClean(headers)
    C->>P: start curl -q -sS --connect-timeout 30 --speed-time N --speed-limit 1 -K - -D - --suppress-connect-headers -- url
    C->>P: stdin: header = "Name: value" lines, then close
    loop while status is 1xx
        P-->>C: status line, header lines until blank line
        C->>C: parse status, lowercase header names into Map
    end
    C-->>Caller: Response with body = FilterInputStream over stdout
    Caller->>C: body.read() returns -1
    C->>P: waitFor 30 s, check exitValue
    alt exit != 0
        C-->>Caller: IOException with curl's stderr
    end
    Caller->>C: close()
    C->>P: destroy() then close stream
```

- Headers are delivered by `-K -` (a curl config block on stdin), quoted with `\` and `"` escaped, so tokens are
  never on the process list. `--` before the URL plus `requireHttp` means the URL can never be an option.
- `-q` ignores the user's `.curlrc`. `-L` is not given: redirects come back as 3xx, like the JDK transport.
- The timeout becomes `--speed-time ceil(timeout/1s)` with `--speed-limit 1`: a stall detector, the only curl
  timeout that also spans the headers.
- The body stream checks the exit code at EOF (`ended()`), so a transfer curl aborted is an `IOException`, not a
  silently truncated file. `close()` destroys the process first, because a reader blocked in the buffered stream
  holds its lock until curl's pipe reaches EOF.
- `failure(Process, URI)` returns curl's stderr text, or "curl exited with <code|no response>" when empty;
  `line(InputStream)` reads one ISO-8859-1 header line and strips CR.

---

## 6. follow: redirects and credential stripping

Default method on `Transport`, shared by both transports.

```mermaid
flowchart TD
    S["follow(uri, headers, timeout)"] --> G["response = get(current, currentHeaders, timeout)"]
    G --> R{"status in 301, 302, 303, 307, 308?"}
    R -- no --> Ret["return response"]
    R -- yes --> L["location = header location; response.close()"]
    L --> M{"location null or redirect == 5?"}
    M -- yes --> E1["IOException: redirect loop or missing Location"]
    M -- no --> N["next = current.resolve(location)<br/>IllegalArgumentException becomes IOException"]
    N --> O{"same scheme, host and port?"}
    O -- yes --> U["current = next"]
    O -- no --> X["copy headers without Authorization"] --> U
    U --> G
```

`jinfer-hub`'s `Fetch.sendOnce` implements the same loop itself (so it can wait on `RateLimit` per hop) rather than
calling `follow`.

---

## 7. deadline: bounding the body

```mermaid
sequenceDiagram
    participant Caller
    participant T as Transport.deadline
    participant D as Transports.DEADLINES
    participant R as inner Response
    Caller->>T: deadline(response, timeout)
    T->>D: schedule(task, timeout)
    T-->>Caller: wrapping Response
    alt caller closes in time
        Caller->>T: close()
        T->>D: future.cancel(false), removed from queue
        T->>R: close()
    else timeout elapses first
        D->>D: start daemon thread fetch-deadline-close
        D->>R: close(), IOException ignored
        Note over Caller,R: a stalled body read now fails instead of hanging
    end
```

The close runs on its own thread so that a close which waits for the transfer to unwind cannot delay other
deadlines on the single scheduler thread.

---

## 8. Proxies: environment proxy selection

```mermaid
flowchart TD
    A["Proxies.selector()"] --> B["selector(System::getenv)"]
    B --> C["https = https_proxy, else HTTPS_PROXY<br/>http = http_proxy only, lowercase"]
    C --> D{"both null?"}
    D -- yes --> E["ProxySelector.getDefault()<br/>keeps -Dhttps.proxyHost working"]
    D -- no --> F["new EnvProxySelector(http, https, no_proxy or NO_PROXY)"]
    F --> G["select(uri)"]
    G --> H{"host matches a NO_PROXY entry?<br/>star, exact, or subdomain; leading dot ignored"}
    H -- yes --> I["Proxy.NO_PROXY"]
    H -- no --> J["https scheme: https proxy, else http proxy;<br/>NO_PROXY when that one is null"]
```

- Blank values count as unset; values are stripped. `HTTP_PROXY` (uppercase) is ignored, as curl does (httpoxy,
  CVE-2016-5385), so the JVM and the curl-based native image route the same way.
- `parse` adds `http://` when no scheme is present, defaults the port to 8080, and returns `null` on a malformed
  value so a broken variable never blocks direct access. `connectFailed` does nothing.
- `selector(UnaryOperator getenv)` is the test seam (`ProxiesTest`).

---

## 9. Hardening and native-image notes

| Concern | Where | Mechanism |
|---|---|---|
| Non-http URL / option injection | `Transports.requireHttp`, `CurlTransport` | scheme must be http/https with a host; `--` before URL |
| Header smuggling | `Transports.requireClean` | null/blank name, `:` in name, CR/LF in name or value → `IOException` |
| Token on process list | `CurlTransport` | headers via `-K -` on stdin |
| Token leaked to CDN | `Transport.follow` | `Authorization` dropped on scheme/host/port change |
| Truncated download | `CurlTransport` body | exit code checked at EOF |
| Hung body | `Transport.deadline` | scheduled forced close |
| Build-time heap capture | `native-image.properties` | `--initialize-at-run-time=com.qxotic.jinfer.fetch` (transport choice, `HttpClient`, deadline thread are made at run time) |
| No TLS in native image | `jinfer-native/pom.xml` | `JdkTransport` provider excluded → curl |

Tests (`src/test`): `TransportTest` and `TransportQaTest` run both transports against a local `jdk.httpserver`
server (hence the `--add-modules jdk.httpserver --add-reads` in the compiler and surefire config); `ProxiesTest`
covers variable precedence and the leading-dot `NO_PROXY` spelling.
