# QUANT PARADISE 
![GitHub](https://img.shields.io/github/license/Naman-Goyal/COMS4995?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Naman-Goyal/COMS4995?logo=Github)
[![Build Status](https://travis-ci.org/Naman-Goyal/Quant-paradise.svg?branch=master)](https://travis-ci.org/Naman-Goyal/Quant-paradise)
[![codecov](https://codecov.io/gh/Naman-Goyal/Quant-paradise/branch/master/graph/badge.svg?token=A04BI6DMPU)](https://codecov.io/gh/Naman-Goyal/Quant-paradise)
[![Documentation Status](https://readthedocs.org/projects/quant-paradise/badge/?version=latest)](https://quant-paradise.readthedocs.io/en/latest/?badge=latest)

Link to  repository:[Here](https://github.com/harshhacks/quantparadise)
Looking to create a utility/tool to help Portfolio Managers in data aquisition and modelling.Currently plan to create a data acquisition tool to obtain data from popular platforms(preferably Yahoo Finance), and use the CIR and Vasicek models for modeling interest rates and zero-coupon bond prices over different time horizons and calibrate them. Will also provide graphing modules using pre-existing open source tools to analyze the models and generate mock portfolios and calculate and plot relevant metrics.


## Examples/Demonstration:

The below images show a few images screenshot from our Jupyter file which demonstrate how to use the library. First, we need to initialize the values of the LIBOR rates and Swap rates for any two dates. In the future, these can be read using a CSV file, but for now, we initialize them as numpy arrays below. Each array is comprised of lists. 

For LIBOR rates, the first value of each list represents the time period. For instance, 1/12 would represent the 1 month rate, 12/12 would represent the 1 year rate and so on. 
The second value represents the value of that interest rate on the first date, and the third value represents the value of the interest rate for the second date. 

For SWAP rates, the first value represents the duration of the swap rate. The other two values are used in the same way as the LIBOR rates 


![alt text](https://github.com/Naman-Goyal/Quant-paradise/blob/master/1.PNG?raw=true)
The image below shows interpolation of rates between the two dates. For this example, we calibrate on half-day periods but these can be adjusted according to need and precision. 

![alt text](https://github.com/Naman-Goyal/Quant-paradise/blob/master/2.PNG?raw=true)

Below we show examples of the graphs obtained for the Vasicek and CIR calibrations for the two dates we picked:
![alt text](https://github.com/Naman-Goyal/Quant-paradise/blob/master/VASEICK_first_date.png?raw=true)
![alt text](https://github.com/Naman-Goyal/Quant-paradise/blob/master/CIR_first_date.png?raw=true)

Finally, we show a graph showing the price of a zero coupon bond using the above calibration. 
![alt text](https://github.com/Naman-Goyal/Quant-paradise/blob/master/CIR_results_comparison.png?raw=true)



