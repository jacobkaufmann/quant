# quant
A repository dedicated to ML-based quantitative finance

## Collect equity data
#### Comma separated list:
```
python get_equity_data.py --equities=AAPL,IBM,FB
```
#### All available US equities on Nasdaq, NYSE, and AMEX:
```
python get_equity_data.py --all_equities=yes
```
## Process equity data
#### Comma separated list:
```
python process_equity_data.py --equities=AAPL,IBM,FB
```
#### All files in pre-processed data directory:
```
python process_equity_data.py --all_equities=yes
```
## Train and Test DNN
```
python quant.py
```
