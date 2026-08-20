from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

def setup_telemetry(service_name: str = "enterprise-rag-platform"):
    """Configures OpenTelemetry tracer provider."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

tracer = setup_telemetry()
