## Files

* **train.csv** - The training set
  * `timestamp` - A timestamp for the minute covered by the row.
  * `Asset_ID` - An ID code for the cryptoasset.
  * `Count` - The number of trades that took place this minute.
  * `Open` - The USD price at the beginning of the minute.
  * `High` - The highest USD price during the minute.
  * `Low` - The lowest USD price during the minute.
  * `Close` - The USD price at the end of the minute.
  * `Volume` - The number of cryptoasset units traded during the minute.
  * `VWAP` - The volume weighted average price for the minute.
  * `Target` - 15 minute residualized returns. See [the &#39;Prediction and Evaluation&#39; section of this notebook](https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition) for details of how the target is calculated.
* **example_test.csv** - An example of the data that will be delivered by the time series API.
* **example_sample_submission.csv** - An example of the data that will be delivered by the time series API. The data is just copied from  **train.csv** .
* **asset_details.csv** - Provides the real name and of the cryptoasset for each `Asset_ID` and the weight each cryptoasset receives in the metric.
* **gresearch_crypto** - An unoptimized version of the time series API files for offline work. You may need Python 3.7 and a Linux environment to run it without errors.
* **supplemental_train.csv** - After the submission period is over this file's data will be replaced with cryptoasset prices from the entire submission period. The current copy has been updated from the original and covers roughly 2/3 of the submission period.

### Time-series API Details

* Refer to [the time series introduction notebook](https://www.kaggle.com/sohier/detailed-api-introduction) for an example of how to complete a submission. The time-series API has changed somewhat from previous competitions!
* Expect to see roughly three months worth of data in the test set. Until the forecasting phase of the competition, the API will just deliver a slice of the training data.
* The API will require 0.5 GB of memory after initialization. The initialization step (`env.iter_test()`) will require meaningfully more memory than that; we recommend you do not load your model until after making that call. The API will also consume less than 30 minutes of runtime for loading and serving the data.
* The API loads the data using the following types:
  Asset_ID: int8, Count: int32, row_id: int32, Count: int32, Open: float64, High: float64, Low: float64, Close: float64, Volume: float64, VWAP: float64
