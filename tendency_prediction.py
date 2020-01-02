"""get_TypeL.py: Get a type label of Brexit events from exchange rate changing tendency."""
__author__ = "Siyuan Liu"


# Try LSTM and NN with other structure.
# Set a prediction starts date. ✅
# Implement a validation approach.✅
# Load real-time data from internet. ✅
# Tackle the problem that dates in future can contain currency market holidays and weekend days inside. ✅


""" Global configuration variables """

# Patience of earlystopping.
from decimal import Decimal
import datetime as dt
import numpy as np
import requests
import time
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
from cwrnn import ClockworkRNN
from keras.datasets.imdb import load_data
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import backend as K
from pandas import DataFrame
import pandas as pd
patience = 32
# Epochs of model learning.
epochs = 512
# Defind global loop time, one round will take about 1 min.
loop_time = 1024
# 'Refresh Times': Define how many times that the error needs to be refreshed(diminished) before iteration stopped.
# 8 is totally enough.
refresh_times = 16
# Define the dates' length that are going to predict.
predict_Length = 32
# The exchange rate brfore '2016-7-13' is too special.
initial_date = pd.Timestamp('2016-7-13')
# Set a date and let the prediction starts from that day.
# The last day so far is 2019-11-29.
predict_date = pd.Timestamp('2019-11-20')


""" if there are any numbers inside a str, return True, or return False. """


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


""" Send request to get the datasets of exchange rate in time order of US to UK and US to EU,
then generate a dataset of UK to EU """


def getData_FromWeb():
    url_US2UK = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DEXUSUK&scale=left&cosd=2014-11-29&coed=2019-11-29&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date=2019-12-03&revision_date=2019-12-03&nd=1971-01-04"
    url_US2EU = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DEXUSEU&scale=left&cosd=2014-11-29&coed=2019-11-29&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date=2019-12-03&revision_date=2019-12-03&nd=1999-01-04"
    webGet_US2UK = requests.get(url_US2UK).content
    webGet_US2EU = requests.get(url_US2EU).content
    # Columns of data_US2UK are 'DATE' and 'DEXUSUK'.
    data_US2UK = pd.read_csv(io.StringIO(webGet_US2UK.decode('utf-8')))
    # Columns of data_US2EUR are 'DATE' and 'DEXUSEU'.
    data_US2EU = pd.read_csv(io.StringIO(webGet_US2EU.decode('utf-8')))
    data_US2UK['DATE'] = pd.to_datetime(data_US2UK['DATE'])
    data_US2EU['DATE'] = pd.to_datetime(data_US2EU['DATE'])
    date_number = -1
    invalid_datesA = 0
    for date in data_US2UK['DATE']:
        date_number += 1
        if date.__ne__(data_US2EU['DATE'][date_number]):
            invalid_datesA += 1
    iRATE = pd.DataFrame()
    rate_list = []
    if invalid_datesA == 0:
        date_number = -1
        for rate in data_US2UK['DEXUSUK']:
            date_number += 1
            if hasNumbers(rate):
                if hasNumbers(data_US2EU['DEXUSEU'][date_number]):
                    rate_list.append(
                        round((Decimal(str(rate))/Decimal(str(data_US2EU['DEXUSEU'][date_number]))), 5))
                else:
                    rate_list.append(0)
            else:
                rate_list.append(0)
        iRATE['DATE'] = data_US2UK['DATE']
        iRATE['RATE'] = rate_list
    return iRATE


""" Search the date of every Brexit event, relate it to the corresponding exchange rate BY THE DATE.
The dates in exchange rate dataset are strictly in time order whereas the dates of Brexit events doesn not. """


def getDate_fromER(date):
    date_number = -1
    # Initialise default first and last two dates.
    # This is the first day in the dataset.
    l_date = dt.date(2016, 1, 1)
    g_date = dt.date.today()
    lDate_number = gDate_number = 0
    for date_pointer in RATE['DATE']:
        date_number += 1
        # If current date is less than or equal to event date.
        # Get and return the left and right dates that are adjacent to it.
        if date_pointer.__lt__(date):
            l_date = date_pointer
            lDate_number = date_number
        if date_pointer.__gt__(date):
            gDate_number = date_number
            break
    # Only return the number of both dates in 'newRATE['DATE']'.
    # Warning: the number of the Brexit date isn't necessary to be the median of lDate_number and gDate_number.
    # Because in some situations the date of the Brexit event might not exist in the exchange rate date list.
    return lDate_number, gDate_number


""" For thoes points which are during rest days, generating rates for them by deviding points by fixed proportionality """


def generate_Rate(be_date):
    # Get the 'lDate' and 'gDate' from the date in exchange rate list.
    lDate_number, gDate_number = getDate_fromER(be_date)
    # 'lDate' is the closest one in the dates less than our goal date.
    # 'gDate' is the closest one in the dates greater than our goal date.
    lDate = RATE['DATE'][lDate_number]
    gDate = RATE['DATE'][gDate_number]
    # Caculate the space between the date of Brexit event and 'lDate' and 'gDate'.
    lSpace = gSpace = 0
    while lDate.__lt__(be_date) is True:
        lSpace += 1
        lDate = lDate + time_unit
    while gDate.__gt__(be_date) is True:
        gSpace += 1
        gDate = gDate - time_unit
    # Devide points by fixed proportionality.
    coefficient = Decimal(str(lSpace))/Decimal(str(gSpace))
    return (Decimal(str(RATE['RATE'][lDate_number]))+coefficient*Decimal(
        str(RATE['RATE'][gDate_number])))/(Decimal(1)+coefficient)


""" Judge if or not this input date exists in the exchange rate list.
If exist, ruturn True and date_number, else return False and generated rate. """


def if_DateExist(goal_date):
    date_number = -1
    for er_date in RATE['DATE']:
        date_number += 1
        if er_date.__eq__(goal_date):
            return True, date_number
    return False, generate_Rate(goal_date)


""" Encapsulate data into two new but practical dataset, 'er_data' and 'be_data'. """


def encapsulate_Data():
    # Initialise the loop coefficients.
    be_rate_list = []
    be_number = -1
    # Get the exchange rate value of every event.
    for be_date in EVENTS['DATE']:
        be_number += 1
        # If this date from a brexit event also exists in exchange rate date list, use the original value directly.
        # Or caculate a rate value for this brexit event.
        if_exist, dateOr_rate = if_DateExist(be_date)
        if if_exist:
            be_rate_list.append(RATE['RATE'][dateOr_rate])
        else:
            inGapP_list.append(be_number)
            be_rate_list.append(dateOr_rate)
    # Combine data.
    be_data = pd.DataFrame()
    be_data['RATE'] = be_rate_list
    be_data['DATE'] = EVENTS['DATE']
    return be_data


""" There will be currency market holidays and weekend days inside the datelist,
place them at the valid positions """


def optimize_Datelist(goal_datelist):
    # Convert all date-strings in 'goal_datelist' into datetime.
    date_number = -1
    for date in goal_datelist:
        date_number += 1
        goal_datelist[date_number] = dt.datetime.strptime(
            ((str(date)).split())[0], '%Y-%m-%d')
    len_gdl = len(goal_datelist)
    date_number = -1
    while date_number < len_gdl - 1:
        date_number += 1
        date_number_i = date_number
        datew = goal_datelist[date_number].weekday()
        # If the day is Saturday.
        if datew == 5:
            for date_i in goal_datelist[date_number:]:
                goal_datelist[date_number_i] = date_i+time_unit+time_unit
                date_number_i += 1
        # If the day is Sunday.
        elif datew == 6:
            for date_i in goal_datelist[date_number:]:
                goal_datelist[date_number_i] = date_i+time_unit
                date_number_i += 1
        # Refresh 'datew' because 'goal_datelist' has been refreshed.
        datew = goal_datelist[date_number].weekday()
        date_number_ii = date_number
        # If this is a currency market holiday, and isn't weekend, push one day ahead again.
        try:
            bool, temp_var = if_DateExist(goal_datelist[date_number])
        except Exception:
            pass
        else:
            # If no exception, it means this date isn't belong to the future.
            if bool:
                pass
            else:
                if datew == 5 or datew == 6:
                    pass
                else:
                    for date_i in goal_datelist[date_number:]:
                        goal_datelist[date_number_ii] = date_i+time_unit
                        date_number_ii += 1
    return goal_datelist


""" Generate future date list according to the future rate list. """


def generate_FutureD(future_rate):
    # Create a list contains generated future dates that correspond to every single future rate.
    future_date = []
    i = 0
    for future_datum in future_rate:
        i += 1
        last_date = predict_date
        for _i in range(i):
            last_date = last_date + time_unit
        future_date.append((str(last_date).split())[0])
    # Optimize the date list.
    future_date = optimize_Datelist(future_date)
    return future_date


""" Draw a graph in a global view of exchange rate tedency and show Brexit event on it. """


def draw_erGraph(be_data, future_rate):
    # Initialise the subplot.
    fig, ax = plt.subplots()
    # Highlight points in 'er_data' datalist.
    ax.scatter(RATE['DATE'], RATE['RATE'], color='blue', s=8)
    # # Highlight Brexit events points.
    ax.scatter(be_data['DATE'], be_data['RATE'],
               color='black', s=12)
    # Annotate for every Brexit point.
    for i, number in enumerate(range(len(be_data)-1, -1, -1)):
        ax.annotate(str(number+1),
                    (be_data['DATE'][i], be_data['RATE'][i]), size=8, color='red')
    # Create two lists contain the data to the points during the rest days.
    inGapP_date = []
    inGapP_rate = []
    for inGapP in inGapP_list:
        inGapP_date.append(be_data['DATE'][inGapP])
        inGapP_rate.append(be_data['RATE'][inGapP])
    # Highlight 'in-gap' point.
    ax.scatter(inGapP_date, inGapP_rate, color='red', s=12)
    # Generate future date list according to the future rate list.
    future_date = generate_FutureD(future_rate)
    # DataFrame of 'Future data'.
    fd = pd.DataFrame()
    fd['DATE'] = future_date
    fd['DATE'] = pd.to_datetime(fd['DATE'])
    fd['RATE'] = future_rate
    # Draw predicted points.
    ax.scatter(fd['DATE'], fd['RATE'], color='black', s=16)
    # Draw a global graph by the original 'RATE'.
    RATE['RATE'] = RATE['RATE'].astype(float)
    sns.lineplot(x="DATE",
                 y="RATE",
                 data=RATE, color="dimgray", label="Real curve")
    # Rates list that before the prediction-starts date.
    RATE_copy = pd.DataFrame()
    RATE_copy = RATE
    rate_frame = RATE_copy.set_index(["DATE"], drop=True)
    sorted_rate = rate_frame.sort_index(axis=1, ascending=True)
    rate_list = sorted_rate[['RATE']]
    new_RATE_y = rate_list.loc[:predict_date]
    new_RATE_y = new_RATE_y.values
    new_RATE_y = new_RATE_y.reshape(new_RATE_y.shape[0])
    # Dates list that before the prediction-starts date.
    RATE_copy['RATE'] = RATE_copy['DATE'].values
    date_frame = RATE_copy.set_index(["RATE"], drop=True)
    sorted_date = date_frame.sort_index(axis=1, ascending=True)
    date_list = sorted_date[['DATE']]
    new_RATE_x = date_list.loc[:predict_date]
    new_RATE_x = new_RATE_x.values
    new_RATE_x = new_RATE_x.reshape(new_RATE_x.shape[0])
    # Create a DataFrame contains a datum of the fisrt signle day before the prediction-starts date.
    last_RATE = pd.DataFrame()
    last_RATE['DATE'] = new_RATE_x[-1:]
    last_RATE['RATE'] = new_RATE_y[-1:]
    last_RATE['RATE'] = last_RATE['RATE'].astype(float)
    last_RATE['DATE'] = pd.to_datetime(last_RATE['DATE'])
    # Draw a division vertical line for highlighting the prediction-starts position.
    points_x = [[new_RATE_x[-1:], new_RATE_x[-1:]]]
    points_y = [[1, 1.3]]
    for i in range(len(points_x)):
        plt.plot(points_x[i], points_y[i], color='b')
    # Merge both dataframes for plotting.
    merged_df = pd.concat([last_RATE, fd], ignore_index=True)
    # Plot lines of predicted data.
    merged_df['RATE'] = merged_df['RATE'].astype(float)
    merged_df['DATE'] = pd.to_datetime(merged_df['DATE'])
    sns.lineplot(x="DATE",
                 y="RATE",
                 data=merged_df, color="coral", label="Predicted trend")
    # Show graph with some pre-set layout.
    plt.grid(True)
    plt.tight_layout()
    plt.show()


""" Prepara the data for training. """


def prepare_Data4T():
    # Set the DataFrame rates labels using dates column.
    rate_frame = RATE.set_index(["DATE"], drop=True)
    # Sort data along row or column direction merely according to the ascending order.
    # P.S.'sort_indes' does not have a descending order option.
    sorted_rate = rate_frame.sort_index(axis=1, ascending=True)
    # Merely extract the value list from dataframe.
    rate = sorted_rate[['RATE']]
    # 'loc' works on labels in the index.
    # Extract the training set from the universal set.
    tra_Set = rate.loc[initial_date:predict_date]
    # Extract only values.
    tra_Set = tra_Set.values
    # Define the 'lag' distance between two fllowing lists.
    lag_dis = 1
    # Divide a set of time-sequence single-variable data into two lists which lag in a specified distance with the other one.
    x_train = tra_Set[: -lag_dis]
    y_train = tra_Set[lag_dis:]
    # Reshape the data set as a format fits the requirement of LSTM. i.e.[Sample, Epoch, Feature]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    return x_train, y_train


""" Make prediction according to the best model saved so far.
Return a list contains the predicted rates in the future. """


def make_old_Prediction():
    x_train, y_train = prepare_Data4T()
    es = EarlyStopping(monitor='loss', patience=patience)
    K.clear_session()
    model = Sequential()
    model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128],
                           units_per_period=8,
                           input_shape=(1, 1),
                           output_units=1))
    model.compile(optimizer='adam', loss='mse')
    cwrnn_weights = np.load((path + model_filename), allow_pickle=True)
    model.set_weights(cwrnn_weights)
    # Initialise a list for containing the predicted data in the future.
    future_rate = []
    # 'verbose' is the mode you want to see the feedback content on the console.(0 = silent, 1 = progress bar, 2 = one line per epoch)
    prediction = model.predict(x_train, verbose=0)
    for i in range(predict_Length):
        prediction = prediction.reshape(
            (prediction.shape[0], 1, prediction.shape[1]))
        prediction = model.predict(prediction, verbose=0)
        future_rate.append(
            round(float(((str(prediction[-1:])).strip('[[')).strip(']]')), 5))
    future_rate = np.asarray(future_rate)
    future_rate = future_rate.reshape(future_rate.shape[0])
    return future_rate


""" Make prediction and save the best model according the average model.
Return a list contains the predicted rates in the future. """


def make_new_Prediction():
    x_train, y_train = prepare_Data4T()
    # plot_model(model, to_file='model.png')
    es = EarlyStopping(monitor='loss', patience=patience, mode='auto')
    callbacks_list = [es]
    # Initialise average error variable as 1 or getting the value from saved history.
    if os.path.exists(path + 'best_error.npy'):
        error_ave = np.load((path + 'best_error.npy'), allow_pickle=True)
    else:
        error_ave = 0.1
    refresh_times_i = 0
    iteration_times = 0
    final_future_rate = []
    while True:
        start_Time = time.time()
        K.clear_session()
        model = Sequential()
        model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128],
                               units_per_period=8,
                               input_shape=(1, 1),
                               output_units=1))
        model.compile(optimizer='adam', loss='mse')
        iteration_times += 1
        # Initialise a list for containing the predicted data in the future.
        future_rate = []
        # 'verbose' is the mode you want to see the feedback content on the console.(0 = silent, 1 = progress bar, 2 = one line per epoch)
        history = model.fit(x=x_train, y=y_train, verbose=0,
                            epochs=epochs, callbacks=callbacks_list)
        prediction = model.predict(x_train, verbose=0)
        for i in range(predict_Length):
            prediction = prediction.reshape(
                (prediction.shape[0], 1, prediction.shape[1]))
            prediction = model.predict(prediction, verbose=0)
            future_rate.append(
                round(float(((str(prediction[-1:])).strip('[[')).strip(']]')), 5))
        future_rate = np.asarray(future_rate)
        future_rate = future_rate.reshape(future_rate.shape[0])
        error_ave_i = evaluate_PerOfPre(future_rate)
        if error_ave_i*error_ave_i < error_ave*error_ave:
            final_future_rate = future_rate
            error_ave = error_ave_i
            refresh_times_i += 1
            # Save the best error so far.
            np.save((path + 'best_error.npy'), error_ave)
            # Save weights as ndarray file.
            weights = np.asarray(model.get_weights())
            np.save((path + model_filename), weights)
            print('Better model saved, in the iteration times at ' + str(iteration_times) +
                  ' and refresh times ' + str(refresh_times_i) + '/' + str(refresh_times) + ' with average error: ' + str(error_ave))
            continue
        if refresh_times_i >= refresh_times:
            print('Refreshing of ' + str(refresh_times_i) + ' times has finished after' +
                  str(iteration_times) + ' rounds, the final average error is: ' + str(error_ave))
            break
        if iteration_times >= loop_time:
            print('Iteration of ' + str(iteration_times) + ' rounds finished, the least average error so far is: ' +
                  str(error_ave) + ', after error has been refreshed ' + str(refresh_times_i) + ' times.')
            break
        end_Time = time.time()
        print('Time use: ' + str(end_Time-start_Time) + ' , finished round ' +
              str(iteration_times) + '/' + str(loop_time) + ' while error has been refreshed for ' + str(refresh_times_i) + ' times, with value ' + str(error_ave))
        # # Plot training & validation loss values.
        # plt.plot(history.history['loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.show()
    return final_future_rate


""" Evaluate the performance of prediction, return a value of average error. """


def evaluate_PerOfPre(future_rate):
    # Generate future date list according to the future rate list.
    future_date = generate_FutureD(future_rate)
    # DataFrame of 'Future data'.
    fd = pd.DataFrame()
    fd['DATE'] = future_date
    fd['DATE'] = pd.to_datetime(fd['DATE'])
    fd['RATE'] = future_rate
    RATE_copy = pd.DataFrame()
    RATE_copy = RATE
    rate_frame = RATE_copy.set_index(["DATE"], drop=True)
    sorted_rate = rate_frame.sort_index(axis=1, ascending=True)
    rate_list = sorted_rate[['RATE']]
    last_future_date = ((future_date[-1:])[0]).strftime("%Y-%m-%d")
    last_future_date = pd.Timestamp(last_future_date)
    real_rate = rate_list.loc[predict_date:last_future_date]
    real_rate = real_rate.values
    real_rate = real_rate.reshape(real_rate.shape[0])
    # real_rate = real_rate[1:]
    e_total = 0
    rate_number = -1
    if real_rate[1:].shape[0] != 0:
        for rr in real_rate[1:]:
            rate_number += 1
            e_total = e_total + (rr - future_rate[rate_number])
        # 0.2 is a coefficient of proportion, to amplify error because of its own scale.
        e_average = e_total/(rate_number+1)
        return e_average
    else:
        return 0


# For machine learning.
# from keras.utils import plot_model
# A Clockwork RNN (Jan Koutník 2014)
# Extended ploting lib.
# Libs for sending interent request.
# Extended datetime var-type integrated lib.
# Defimal precise calculation lib.
# # Print information without abbreviation.
# np.set_printoptions(threshold=np.inf)


""" Global Pre-processing"""


# Read data from csv file.
iRATE = getData_FromWeb()
# Delete the dates which are with invalid(zero) exchange rate vaule.
row_number = -1
for rate in iRATE['RATE']:
    row_number += 1
    if round(rate) == 0:
        iRATE = iRATE.drop(index=[row_number])
# Refresh the index of new dataframe. (Or the deleted location will have a blank)
RATE = pd.DataFrame()
RATE['DATE'] = np.array(iRATE['DATE'])
RATE['RATE'] = np.array(iRATE['RATE'])
RATE['RATE'] = RATE['RATE'].astype(float)
# Define root path of project folder.
path = "/Users/liusiyuan/Desktop/Codes/PredictExchangeRate_According_BrexitEvent/"
# Define the name of model file.
model_filename = "cwrnn_weights.npy"
be_file = "EVENTS.csv"
# Define a global list for containing the Brexit events point in the rest days.
inGapP_list = []
# Define a global single day date unit.
time_unit = dt.timedelta(days=1)
# In case of this date is invalid as a coincidence.
for date in [initial_date, predict_date]:
    bool_i, temp_var = if_DateExist(date)
    if bool_i:
        pass
    else:
        lDate_number, gDate_number = getDate_fromER(date)
        if date.__eq__(initial_date):
            initial_date = RATE['DATE'][gDate_number]
        if date.__eq__(predict_date):
            predict_date = RATE['DATE'][gDate_number]
# Get EVENTS as DataFrame.
EVENTS = pd.read_csv(path + be_file)
EVENTS = pd.DataFrame(EVENTS, columns=['DATE', 'EVENT'])
# Convert the date from string type to the 'datetime'.
EVENTS['DATE'] = pd.to_datetime(EVENTS['DATE'])


def main():
    future_rate = make_old_Prediction()
    be_data = encapsulate_Data()
    draw_erGraph(be_data, future_rate)


if __name__ == '__main__':
    main()
