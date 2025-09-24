import logging
import pytest

def test_no_pii_in_logs(caplog):
    logger = logging.getLogger("app")
    with caplog.at_level(logging.INFO):
        logger.info("user requested prediction", extra={"country": "US"})
    # Example policy: ensure keywords like "ssn" not present
    banned = ["ssn", "password"]
    for record in caplog.records:
        msg = record.getMessage().lower()
        assert not any(b in msg for b in banned)
