import json

from creditml.data import generator, loaders


def test_manifest_matches_generator():
    man = json.loads((generator.ROOT / "data" / "manifest.json").read_text())
    a = loaders.applications()
    assert len(a) == man["applications"] and int(a["default_12m"].notna().sum()) == man["mature"]


def test_outcome_window_and_traps():
    a = loaders.applications()
    assert a.loc[a["disbursal_month"] > "2025-06-01", "default_12m"].isna().all()           # immature
    assert a["bureau_score"].isna().any()                                                   # new to credit
    late, early = a[a["disbursal_month"] >= "2025-01-01"], a[a["disbursal_month"] < "2025-01-01"]
    assert (late["channel"] == "digital_partner").mean() > 3 * (early["channel"] == "digital_partner").mean()


def test_lgd_is_bimodal():
    lo = loaders.losses()
    assert (lo["lgd"] < 0.1).mean() > 0.2 and (lo["lgd"] > 0.7).mean() > 0.2
