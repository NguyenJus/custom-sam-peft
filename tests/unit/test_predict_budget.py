def test_decide_predict_budget_warning_over_budget():
    from custom_sam_peft.predict.budget import PREDICT_8GB_BUDGET_GB, decide_predict_budget_warning

    over = int((PREDICT_8GB_BUDGET_GB + 1.0) * 1024**3)
    warn, msg = decide_predict_budget_warning(measured_bytes=over, budget_gb=PREDICT_8GB_BUDGET_GB)
    assert warn is True
    assert "may not be usable" in msg
    assert "8 GB" in msg or "7.0" in msg


def test_decide_predict_budget_warning_under_budget():
    from custom_sam_peft.predict.budget import PREDICT_8GB_BUDGET_GB, decide_predict_budget_warning

    under = int((PREDICT_8GB_BUDGET_GB - 1.0) * 1024**3)
    warn, _msg = decide_predict_budget_warning(
        measured_bytes=under, budget_gb=PREDICT_8GB_BUDGET_GB
    )
    assert warn is False
