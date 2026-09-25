import pytest

from pace_s.config import load_campaign


@pytest.fixture(scope="session")
def cfg_lis():
    return load_campaign("campaign_LiS.yaml")


@pytest.fixture(scope="session")
def cfg_lispan():
    return load_campaign("campaign_LiSPAN.yaml")
