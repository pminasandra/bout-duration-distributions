# Pranav Minasandra
# pminasandra.github.io
# Dec 30, 2022

def bayes_classify(dataframe):
    
    predictions = dataframe.copy()
    predictions["state"] = "A"
    predictions.loc[predictions["feature"] > 0.0, 'state'] = "B"

    return predictions

