"""Stage failure signalling. Stages raise `StageFailure`; the pipeline turns it
into a `StageError` on the response and keeps the earlier stages' output (FR7)."""

from likho.schemas import Stage, StageError


class StageFailure(Exception):
    def __init__(self, stage: Stage, message: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.error = StageError(stage=stage, message=message, recoverable=recoverable)
