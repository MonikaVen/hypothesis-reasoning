"""Protocol failures. These are harness errors, not scored agent mistakes."""


class ProtocolError(Exception):
    """The run or the corpus violates a frozen experimental rule."""


class LessonLeak(ProtocolError):
    """A lesson cites a case outside the training partition."""


class CapabilityError(ProtocolError):
    """A condition was given a tool, lesson, or fact it is not allowed to have."""


class InvalidVariant(ProtocolError):
    """A recorded-data transform does not meet the eligibility rules."""


class CounterfactualRequiresRegeneration(ProtocolError):
    """V3 cannot be built by editing an existing recording."""


class UnsealedProtocol(ProtocolError):
    """Confirmatory execution was requested before the protocol was sealed."""


class InfrastructureError(Exception):
    """The tool host failed. Distinct from an agent-produced error."""
