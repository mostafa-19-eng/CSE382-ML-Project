# Clustering Baseline (Milestone 1)

## Goal

In this part I built a first clustering experiment on the City Hotel data (H2) to see whether bookings fall into natural groups. It is only a baseline: one algorithm (K-Means) with no tuning. The clustering is fully unsupervised. The cancellation label is never used to form the groups or to choose any setting.

## Data

I used `data/H2_cleaned.csv`, which has 78,091 bookings. This is H2.csv (79,330 rows) after the cleaning step, which removed rows with ADR <= 0 or ADR > 1,000 and rows with zero guests. So the results apply to this cleaned data, not the raw file.

I did not split the data into train and test. There is no label to predict, so there is nothing to protect, and the scaling and encoding are fitted on all rows.

## Feature selection

I went through every column in the cleaned file and decided whether it belongs in the clustering. I wanted features that describe the booking and that are known when the booking is made.

| Column | Decision | Reason |
|---|---|---|
| IsCanceled | exclude | this is the outcome |
| LeadTime | include (log1p + scale) | how far ahead the booking was made |
| ArrivalDateYear | exclude | only 3 years (2015-2017), little use for segments |
| ArrivalDateMonth | include (sin/cos) | seasonality, encoded as a circle |
| ArrivalDateWeekNumber | exclude | redundant with month |
| ArrivalDateDayOfMonth | exclude | little clustering signal |
| StaysInWeekendNights | include (scale) | trip length |
| StaysInWeekNights | include (log1p + scale) | trip length |
| Adults | exclude | covered by TotalGuests |
| Children | exclude | covered by TotalGuests |
| Babies | exclude | covered by TotalGuests |
| Meal | include (one-hot, 4 categories) | chosen at booking |
| Country | exclude | 167 values, one-hot would dominate the distance |
| MarketSegment | include (one-hot) | where the booking came from |
| DistributionChannel | include (one-hot) | booking channel |
| IsRepeatedGuest | include (0/1) | returning guest or not |
| PreviousCancellations | include (log1p + scale, borderline) | guest history, may relate to the outcome |
| PreviousBookingsNotCanceled | include (log1p + scale) | guest history |
| ReservedRoomType | exclude | 7 values, low value for segmentation |
| AssignedRoomType | exclude | decided after booking |
| BookingChanges | exclude | grows after booking |
| DepositType | exclude | indirect proxy for the outcome (see below) |
| Agent | exclude | 223 ID numbers, not a real category |
| Company | exclude | 201 ID numbers, same problem |
| DaysInWaitingList | include (log1p + scale) | demand signal |
| CustomerType | include (one-hot) | transient, contract, group |
| ADR | exclude | this is the regression target, kept out of the clustering |
| RequiredCarParkingSpaces | include (log1p + scale, borderline) | treated as part of the stored booking record |
| TotalOfSpecialRequests | include (log1p + scale, borderline) | treated as part of the stored booking record |
| TotalGuests | include (scale) | group size |
| RoomMismatch | exclude | built from AssignedRoomType, so it is post-booking |

In total I use 15 raw features (7 log1p + scale, 2 scale only, 4 one-hot, 1 binary, 1 month). After preprocessing there are 30 columns: 7 + 2 + 18 one-hot + 1 + 2 for month.

## Leakage check

- **Direct:** IsCanceled is excluded. ReservationStatus and ReservationStatusDate were already dropped in the cleaning step.
- **Indirect:** DepositType is excluded. 12,867 bookings (16.5% of the data) have a Non Refund deposit, and 99.8% of them were cancelled. If I kept it, the clusters would probably just separate Non Refund bookings, which is almost the outcome itself.
- **After booking:** AssignedRoomType, RoomMismatch and BookingChanges are excluded because they are only known or finished after the booking is made.
- **ID columns:** Country (167), Agent (223) and Company (201) are excluded because one-hot encoding them would create hundreds of columns that dominate the distance.
- **ADR:** excluded because it is the regression target.
- **Borderline:** PreviousCancellations, RequiredCarParkingSpaces and TotalOfSpecialRequests could change during the life of a booking. I kept them and treat each row as the booking record as stored.

## Preprocessing

| Step | Columns | Why |
|---|---|---|
| log1p, then StandardScaler | LeadTime, StaysInWeekNights, PreviousCancellations, PreviousBookingsNotCanceled, DaysInWaitingList, RequiredCarParkingSpaces, TotalOfSpecialRequests | strongly right-skewed (skewness 1.4 to 25.4); log1p shrinks the long tail so a few huge values do not pull the centers |
| StandardScaler | StaysInWeekendNights, TotalGuests | skewness below 1, only need the same scale as the rest |
| OneHotEncoder (min_frequency=0.01) | MarketSegment, DistributionChannel, CustomerType, Meal | turns categories into 18 binary columns; categories under 1% of rows are grouped together |
| passthrough | IsRepeatedGuest | already 0/1 |
| sin/cos (`encode_month` from `src/preprocessing.py`) | ArrivalDateMonth | December and January end up close together |

Whitespace is stripped from all text columns first, because some values like `No Deposit     ` have trailing spaces.

## Algorithm

I used `KMeans(n_clusters=4, n_init=10, random_state=42)` from scikit-learn (k-means++ start by default). K-Means is fast on 78k rows, uses the full dataset, and its cluster centers are easy to read. I chose k=4 as a starting point. I did not pick it with any score, and I did not tune anything in this milestone.

## Results

| Metric | Value |
|---|---|
| Silhouette (10,000-row sample, seed 42) | **0.1246** |
| Davies-Bouldin | **2.0061** |
| Inertia | 687,445.57 |
| Rows / features | 78,091 / 30 |
| Fit time | about 1.1 s |

Cluster sizes: C0 12,792 (16.4%), C1 36,212 (46.4%), C2 25,771 (33.0%), C3 3,316 (4.2%).

The Silhouette is computed on a sample because it needs distances between all pairs of rows, which is slow on the full data. Davies-Bouldin uses all rows.

## What the scores mean

Both scores say the separation is weak (Silhouette 0.12, Davies-Bouldin 2.0). The likely reasons:
- K-Means uses Euclidean distance, which treats 18 binary one-hot columns the same way as scaled continuous columns.
- The clusters may not be round or similar in size, which is what these scores prefer.
- k=4 was not optimized.

The PCA plot agrees: the clusters overlap a lot in 2D (the first two components explain 14.7% and 12.9% of the variance).

## Clusters

These numbers come from the table the script prints (mean of each numeric feature per cluster). I describe only what the numbers show, not why customers behave this way.

- **C0** (n=12,792): lead time 10 days (111 overall), PreviousBookingsNotCanceled 0.68 (0.11 overall), waiting list about 0.
- **C1** (n=36,212): lead time 153 days, PreviousBookingsNotCanceled about 0, special requests 0.05 (0.55 overall).
- **C2** (n=25,771): lead time 92 days, special requests 1.35 (0.55 overall), TotalGuests 2.3 (2.0 overall), 2.7 week-nights.
- **C3** (n=3,316): lead time 192 days, days in waiting list 76.8 (3.3 overall), PreviousCancellations 0.30 (0.08 overall).

## Check against the outcome (after clustering only)

After clustering, I compared the groups with the cancellation label, ADR and deposit type, which the clustering never saw.

| Cluster | Cancel rate | Non Refund share | Mean ADR |
|---|---|---|---|
| C0 | 16.6% | 1.9% | 105 |
| C1 | 59.2% | 29.4% | 98 |
| C2 | 27.7% | under 0.1% | 122 |
| C3 | 68.2% | 58.8% | 91 |
| Overall | 42.0% | 16.5% | - |

The cancellation rate differs a lot between clusters (16.6% to 68.2%), even though IsCanceled and DepositType were not used. I do not think this is leakage. Lead time, waiting list, previous cancellations and market segment are all known when the booking is made, and they are plausibly related to cancellation, so the clusters pick that up on their own. PreviousCancellations stays marked as borderline.

## How to run

1. Install pandas, numpy, scikit-learn and matplotlib (and pytest for the test).
2. Make sure `data/H2_cleaned.csv` exists. If not, run `python src/data_cleaning.py` from the project root.
3. From the project root run `python -m src.clustering.clustering_baseline`.
4. The script prints the metrics, the cluster means and the check against the outcome, and saves these files in `outputs/clustering/`:
   - `baseline_results.json` (metrics and settings)
   - `cluster_labels.csv` (cluster of each row)
   - `pca_2d_clusters.png` (2D view of a 10k sample with the cluster centers)
   - `cluster_profile_heatmap.png` (numeric feature means per cluster as z-scores)
5. To run the test: `python -m pytest tests/test_clustering.py`. It checks that no excluded column is used and that the labels are identical across two runs.

## For the progress report

Use the results table, the clusters section, the leakage check, and the two figures above. The preliminary clustering leakage audit is the "Leakage check" section.

## Next (Milestone 2)

- Add two more algorithms (for example DBSCAN, GMM or agglomerative) and compare each with its tuned version.
- Choose k and the other settings using only unsupervised scores, never the cancellation label.
- Add a dendrogram on a sample of the data.
- Try to reduce the effect of the one-hot columns, for example with dimensionality reduction.