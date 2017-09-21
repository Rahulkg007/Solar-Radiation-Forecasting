## Load Libraries
require(TSA)
require(forecast)
require(ggplot2)

require(dLagM) # Distributed lag models library

## Set Working Directory
setwd("C:/Users/rahul/Google Drive/RMIT/Semester 2/Forecasting/Assignment 2")

## Import File

# Data About monthly average horizontal solar radiation and the monthly precipitation
# January 1960 and December 2014
Radiation = read.csv('data1.csv')
head(Radiation)

# Conversion to time series Object
Radiation.ts = ts(Radiation,start = 1960,frequency = 12)
Solar.Radiation = ts(Radiation$solar,start = 1960,frequency = 12)
PPT.Radiation = ts(Radiation$ppt,start = 1960,frequency = 12)

##------------
## Preliminary Analysis 
##------------

## Average Solar Radiation
par(mfrow=c(1,1))
plot(Solar.Radiation)
points(x=time(Solar.Radiation),y=Solar.Radiation,pch=as.vector(season(Solar.Radiation)))
# No trend, obvious seasonality, no changing mean but changing variance, Intervention present

par(mfrow=c(1,2))
acf(Solar.Radiation, lag.max = 48, main = "Sample ACF for Average Solar Radiation")
pacf(Solar.Radiation, lag.max = 48, main = "Sample PACF for Average Solar Radiation")
# Indication of Tread and seasonality; Series is stationary

## Check for stationary
adf.test(Solar.Radiation)
# Series is Stationary

## ---------------------------------------
## Forecasting for Average Solar Radiation
##----------------------------------------

# Checking the correlation between Solar and precipitation
# Inverse correlation of 45%
cor(Solar.Radiation,PPT.Radiation)

##
## Distributed Lag Models
##

DLM.Q = finiteDLMauto(x = as.vector(Solar.Radiation) , y = as.vector(PPT.Radiation))

model1 = dlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation) , q = DLM.Q$q , show.summary = TRUE)
checkresiduals(model1$model$residuals)
qqnorm(model1$model$residuals); qqline(model1$model$residuals)
shapiro.test(model1$model$residuals)
# Normality assumption failed

VIF.model = vif(model1$model) 
VIF.model > 10
# Model does not have multi collinearity issue
# Adjusted R square value is very less


##
## Polynomial Distributed Lags
##

model2 = polyDlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation)  , q = DLM.Q$q , 
                 k = 2, show.beta = FALSE , show.summary = TRUE)

## Koyck Transformation
model3 = koyckDlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation)  , show.summary = TRUE)
checkresiduals(model3$model$residuals)
# Serial correlation exists in the model but over model is significant at p < 0.01 and Adjusted R-squared: 0.7591 

VIF.model3 = vif(model3$model) 
VIF.model3 > 10
# Model is not suffering from Multi collinearity

## Autoregressive Distributed Lag Model

model.11 = ardlDlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation) 
                   , p = 1 , q = 1 , show.summary = TRUE)$model
model.22 = ardlDlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation) 
                   , p = 2 , q = 2 , show.summary = TRUE)$model
model.33 = ardlDlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation) 
                   , p = 3 , q = 3 , show.summary = TRUE)$model

models.AIC = AIC(model.11,model.22,model.33)
models.BIC = BIC(model.11,model.22,model.33)

sortScore(models.AIC, score = 'aic')
sortScore(models.BIC, score = 'bic')

checkresiduals(model.33$residuals)
qqnorm(model.33$residuals); qqline(model.33$residuals)

bgtest(model.33)
# No Serial Correlation upto order 1
Box.test(model.33$residuals)
# The data in independently distributed

model.33 = ardlDlm(x = as.vector(PPT.Radiation), y = as.vector(Solar.Radiation) 
                   , p = 3 , q = 3 , show.summary = TRUE)

# Get predictor series values for forecast
Precipitate = read.csv('data.x.csv')
predictor.Precipitate = as.vector(Precipitate$x)
predictor.length = length(predictor.Precipitate)

forecasts.dlm = dlmForecast(model = model1 , x = predictor.Precipitate , h = predictor.length)$forecasts
predictor.dlm = ts(c(Solar.Radiation, forecasts.dlm),start = 1960,frequency = 12)

forecasts.polydlm = polyDlmForecast(model = model2 , x = predictor.Precipitate , h = predictor.length)$forecasts
predictor.polydlm = ts(c(Solar.Radiation, forecasts.polydlm),start = 1960,frequency = 12)

forecasts.koyckdlm = koyckDlmForecast(model = model3 , x = predictor.Precipitate , h = predictor.length)$forecasts
predictor.koyckdlm = ts(c(Solar.Radiation, forecasts.koyckdlm),start = 1960,frequency = 12)

forecasts.ardldlm = ardlDlmForecast(model = model.33 , x = predictor.Precipitate , h = predictor.length)$forecasts
predictor.ardldlm = ts(c(Solar.Radiation, forecasts.koyckdlm),start = 1960,frequency = 12)

par(mfrow=c(1,1))
{
  plot(predictor.ardldlm,type="l")                       
  lines(predictor.dlm,col="Blue",type="l")
  lines(predictor.polydlm,col="Green",type="l")
  lines(predictor.koyckdlm,col="violet",type="l")
  lines(PPT.Radiation,col="Red",type="l")
  legend("topleft",lty=1, text.width = 16, col=c("black","blue","green",'violet', "red"), 
         c("ARDL","DLM", "Polynomial", "Koyck", "Appropriations"))
}
