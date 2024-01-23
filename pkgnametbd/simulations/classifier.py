# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

def bayes_classify(features):
    
    predictions = features.copy()
    # BROCK OPT > 
    # This line is triggering a future warning,
    # ideally, you should resolve it. If not, suppress
    # so your users are not concerned.
    # <
    predictions.loc[:] = "A"
    predictions[features > 0.0] = "B" #WLOG

    return predictions