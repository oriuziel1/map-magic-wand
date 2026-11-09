"""Use the OS certificate store (needed behind TLS-inspecting corporate proxies)."""
import ssl

try:
    import truststore

    truststore.inject_into_ssl()
    SSL_CONTEXT: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except ImportError:  # pragma: no cover
    SSL_CONTEXT = True
