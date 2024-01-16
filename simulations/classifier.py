# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

def bayes_classify(features):
    
    predictions = features.copy()
    predictions.loc[:] = "A"
    predictions[features > 0.0] = "B" #WLOG

    return predictions

