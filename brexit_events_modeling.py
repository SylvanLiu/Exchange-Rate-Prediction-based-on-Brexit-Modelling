import TendencyPredictor as tp
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker

print('The last day has been considered is ' +
      str(tp.RATE['DATE'][-1:]).split()[1])
be_data = tp.encapsulate_Data()


def calculate_Angle():
    angle_list = []
    i = 0
    len_bedata = len(be_data['DATE'])
    for event_locus in be_data['DATE']:
        event_rate = be_data['RATE'][i]
        lDate_number, gDate_number = tp.getDate_fromER(event_locus)
        lDate = tp.RATE['DATE'][lDate_number]
        gDate = tp.RATE['DATE'][gDate_number]
        lSpace = gSpace = 0
        while lDate.__lt__(event_locus) is True:
            lSpace += 1
            lDate = lDate + tp.time_unit
        while gDate.__gt__(event_locus) is True:
            gSpace += 1
            gDate = gDate - tp.time_unit
        scale_coe = 512
        angle = -1
        lRate = tp.RATE['RATE'][lDate_number]
        gRate = tp.RATE['RATE'][gDate_number]
        if lRate != gRate:
            x_l = -lSpace/scale_coe
            x_g = gSpace/scale_coe
            y_l = lRate - event_rate
            y_g = gRate - event_rate
            a_unknown = y_l/x_l
            x_diff = x_g
            y_diff = a_unknown*x_g
            diff_hypo = abs(y_g-y_diff)
            diff_a = sqrt(x_g**2 + y_diff**2)
            diff_b = sqrt(x_g**2 + y_g**2)
            cosine_diff = (diff_a**2 + diff_b**2 -
                           diff_hypo**2)/(2*diff_a*diff_b)
            angle = np.arccos(cosine_diff)
            angle = (angle/np.pi)*180
        else:
            angle = 90
        # Re-calculate for the situation if angle is equal to 0.
        if angle == 0:
            event_rate = tp.RATE['RATE'][gDate_number]
            gRate = tp.RATE['RATE'][gDate_number+1]
            x_l = -(lSpace+gSpace)/scale_coe
            x_g = 1/scale_coe
            y_l = lRate - event_rate
            y_g = gRate - event_rate
            a_unknown = y_l/x_l
            x_diff = x_g
            y_diff = a_unknown*x_g
            diff_hypo = abs(y_g-y_diff)
            diff_a = sqrt(x_g**2 + y_diff**2)
            diff_b = sqrt(x_g**2 + y_g**2)
            cosine_diff = (diff_a**2 + diff_b**2 -
                           diff_hypo**2)/(2*diff_a*diff_b)
            angle = np.arccos(cosine_diff)
            angle = (angle/np.pi)*180
        print(str(len_bedata-i) + ' ' + str(event_locus) +
              ' ' + str(angle))
        angle_list.append(angle)
        i += 1
    draw_Hist(angle_list)


def calculate_Tenacity_A():
    tena_list = []
    i = 0
    len_bedata = len(be_data['DATE'])
    for event_locus in be_data['DATE']:
        lDate_number, gDate_number = tp.getDate_fromER(event_locus)
        gRate = tp.RATE['RATE'][gDate_number]
        for ii in range(0, len(tp.RATE['RATE'])-gDate_number-1):
            if ((gRate - tp.RATE['RATE'][gDate_number+ii])*(gRate - tp.RATE['RATE'][gDate_number+ii+1])) <= 0:
                break
        # print(str(len_bedata-i) + ' ' + str(event_locus) + ' ' + str(ii))
        tena_list.append(ii)
        i += 1


def calculate_Tenacity_B():
    tena_list = []
    i = 0
    len_bedata = len(be_data['DATE'])
    for event_locus in be_data['DATE']:
        event_rate = be_data['RATE'][i]
        lDate_number, gDate_number = tp.getDate_fromER(event_locus)
        gRate = tp.RATE['RATE'][gDate_number]
        for ii in range(0, len(tp.RATE['RATE'])-gDate_number-1):
            if ((event_rate - tp.RATE['RATE'][gDate_number])*(tp.RATE['RATE'][gDate_number+ii] - tp.RATE['RATE'][gDate_number+ii+1])) <= 0:
                break
        print(str(len_bedata-i) + ' ' + str(event_locus) + ' ' + str(ii))
        tena_list.append(ii)
        i += 1

def draw_Hist(x):
    n_bins = 64
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs[0].hist(x, bins=n_bins)
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    # We can also normalize our inputs by the total number of counts
    axs[1].hist(x, bins=n_bins, density=True)
    # Now we format the y-axis to display percentage
    axs[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.show()


# Draw the distributed graph.


def main():
    # calculate_Angle()
    # calculate_Tenacity_A()
    calculate_Tenacity_B()


if __name__ == '__main__':
    print('Start main function in StrengthCalculator.py')
    main()
