import numpy as np

from helpers.excel import read_timeseries
# from helpers.online_fault import _numeric_cols
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import OneClassSVM

FILEPATH = Path("./helpers/timeseries/distillation_column_timeseries_2025_5min.xlsx")
print("Reading data..")
data = read_timeseries(FILEPATH)
print("Coercing into numeric..")
data = data.select_dtypes(include=[np.number])
print("Numeric data: ", data)

X_train = data.iloc[:70080,]
X_test = data.iloc[70080:,]
print(X_train)
print(X_test)

print("Making pipeline..")
model = make_pipeline(
    StandardScaler(),                 # fit ONLY on X_train via pipeline
    OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
)

print("Fitting..")
model.fit(X_train)

print("Predicting..")
pred = model.predict(X_test)          # +1=inlier, -1=outlier
print("Scoring...")
scores = model.decision_function(X_test)  # higher => more normal (0 is boundary)

outlier_mask = (pred == -1)
outlier_idx = np.where(outlier_mask)[0]
print("Detected outliers:", outlier_idx[:20])


# scores: distance to boundary (0 is the learned decision boundary)
# pred: +1=inlier, -1=outlier
results = pd.DataFrame(
    {
        "score": np.asarray(scores).ravel(),
        "pred": np.asarray(pred).ravel(),
    },
    index=X_test.index,
)
results["is_outlier"] = results["pred"] == -1

# Optional downsampling for faster plotting on very long series
plot_every = max(1, len(results) // 20000)
res_plot = results.iloc[::plot_every]

# Plot novelty score over time with outliers 
plt.figure(figsize=(14, 5))
plt.plot(res_plot.index, res_plot["score"], linewidth=1)
plt.axhline(0.0, linewidth=1)  # boundary at 0
out_plot = res_plot[res_plot["is_outlier"]]
plt.scatter(out_plot.index, out_plot["score"], s=10)
plt.title("One-Class SVM novelty score (decision_function) on test set")
plt.xlabel("time / index")
plt.ylabel("score (higher = more normal)")
plt.tight_layout()

# Plot first column and mark outliers on it
if X_test.shape[1] > 0:
    feature_name = X_test.columns[0]
    feat = X_test[feature_name]
    feat_plot = feat.iloc[::plot_every]

    plt.figure(figsize=(14, 5))
    plt.plot(feat_plot.index, feat_plot.values, linewidth=1)
    # mark outliers at their feature value
    feat_out = feat[results["is_outlier"]]
    feat_out_plot = feat_out.iloc[::plot_every]
    plt.scatter(feat_out_plot.index, feat_out_plot.values, s=10)
    plt.title(f"Feature '{feature_name}' with detected outliers")
    plt.xlabel("time / index")
    plt.ylabel(feature_name)
    plt.tight_layout()

plt.show()

