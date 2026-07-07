try:
    import arviz as az; print("arviz", az.__version__)
except Exception as e:
    print("NO arviz:", e)
