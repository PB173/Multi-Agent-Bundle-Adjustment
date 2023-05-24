##################### Import modules and functions #####################
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

################################             Create functions               ################################


def APE_sigma(type, data_vec, sigma, description, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = [['red', 'green'], ['blue', 'purple']]

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                        markerfacecolor='k', markersize=10, linewidth=0)]

    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse_seq1 = np.zeros((3, len(sigma)))
        ape_rmse_seq2 = np.zeros((3, len(sigma)))
        for i in range(len(data)): # for every value of sigma
            data_sig_i = data[i]
            # data_sig_i_seq1 = data_sig_i[::2]
            # data_sig_i_seq2 = data_sig_i[1::2]
            # print(i)
            # print("data sig i:\n", data_sig_i)
            # print(" ")
            # print("data sig i seq1:\n", data_sig_i[0][::2])
            # print(" ")
            # print("data sig i seq2:\n", data_sig_i[0][1::2])
            # print(" ")
            # print(" ")
            # print(" ")
            # print(" ")

            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse_seq1[j,i] = np.mean(data_sig_i[j][::2]) # mean over all iterations for first sequence         
                ape_rmse_seq2[j,i] = np.mean(data_sig_i[j][1::2]) # mean over all iterations for second sequence   

        # Plot
        ax.plot(sigma, ape_rmse_seq1[0], ':',   color = colors[k][0], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(sigma, ape_rmse_seq1[1], 'x--', color = colors[k][0], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, ape_rmse_seq1[2], 'o--', color = colors[k][0], linewidth = 3, markersize = 10) # Data smoothed

        ax.plot(sigma, ape_rmse_seq2[0], ':',   color = colors[k][1], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(sigma, ape_rmse_seq2[1], 'x--', color = colors[k][1], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, ape_rmse_seq2[2], 'o--', color = colors[k][1], linewidth = 3, markersize = 10) # Data smoothed


    
    legend_elements.append(Line2D([0], [0], linestyle=':', color='k', label='Noisy', linewidth=3))
    
    for i in range(len(data_vec)):
        for j in range(2):
            legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i][j], label = labels[i][j], linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
    ax.set_xticks(sigma)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.05, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.01, sigma[-1] + 0.01)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in rotation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.01, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.01, sigma[-1] + 0.01)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')



    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'APE_rmse_{}_sig_{}'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def APE_sigma_norm(type, data_vec, sigma, description, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """

    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = [['red', 'green'], ['blue', 'purple']]

    ax.plot(np.linspace(sigma[0]-0, sigma[-1]+0.02, len(sigma)),100*np.ones(len(sigma)), 'k:', linewidth = 5)

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                        markerfacecolor='k', markersize=10, linewidth=0)]

    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse_seq1 = np.zeros((3, len(sigma)))
        ape_rmse_seq2 = np.zeros((3, len(sigma)))
        for i in range(len(data)): # for every value of sigma
            data_sig_i = data[i]
            
            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse_seq1[j,i] = np.mean(data_sig_i[j][::2]) # mean over all iterations for first sequence         
                ape_rmse_seq2[j,i] = np.mean(data_sig_i[j][1::2]) # mean over all iterations for second sequence   
       
        # Normalize data
        data_BA_norm_seq1 = ape_rmse_seq1[1]/ape_rmse_seq1[0]*100
        data_smooth_norm_seq1 = ape_rmse_seq1[2]/ape_rmse_seq1[0]*100

        data_BA_norm_seq2 = ape_rmse_seq2[1]/ape_rmse_seq2[0]*100
        data_smooth_norm_seq2 = ape_rmse_seq2[2]/ape_rmse_seq2[0]*100
        
        # Plot
        ax.plot(sigma, data_BA_norm_seq1, 'x--', color = colors[k][0], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, data_smooth_norm_seq1, 'o--', color = colors[k][0], linewidth = 3, markersize = 10) # Data smoothed

        ax.plot(sigma, data_BA_norm_seq2, 'x--', color = colors[k][1], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, data_smooth_norm_seq2, 'o--', color = colors[k][1], linewidth = 3, markersize = 10) # Data smoothed
    
    for i in range(len(data_vec)):
        for j in range(2):
            legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i][j], label = labels[i][j], linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
    ax.set_xticks(sigma)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [%]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in translation: {}, {}'.format(description, 'normalized'),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'unbounded':
            ax.set_xlim(-0.01, sigma[-1]+0.01)
        else:
            ax.set_xlim(-0.05, sigma[-1] + 0.01)
            # ax.legend(handles = legend_elements, labelspacing = 1, fontsize = 15)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in rotation: {}, normalized'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.01, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.01, sigma[-1] + 0.01)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'APE_rmse_{}_sig_{}_norm'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def nbr_iterations_sigma(data_vec, sigma, save=0, save_path = '', scenario = '', title_notes = '', lim = []):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'green', 'orange', 'blue', 'purple']
    colors = ['red', 'blue']

    # Extract data
    for k, data in enumerate(data_vec):
        iterations_unlim = data[:,0,:]
        iterations_lim = data[:,-1,:]

        mean_iter_unlim, mean_iter_lim = [], []
        for i in range(iterations_unlim.shape[0]): # for every value of sigma
            mean_iter_unlim.append(np.mean(iterations_unlim[i]))
            mean_iter_lim.append(np.mean(iterations_lim[i]))
        ax.plot(sigma, mean_iter_unlim, 'x--', color = colors[k], linewidth = 3, markersize = 10) # Unbounded
        ax.plot(sigma, mean_iter_lim, 'o--', color = colors[k], linewidth = 3, markersize = 10) # Bounded
    
    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='Unbounded',
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='Bounded',
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    labels = ['MH01', 'MH04']
    for i in range(len(data_vec)):
        # legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label= labels[i], linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.18, 1), handles = legend_elements, labelspacing = 1, fontsize = 15)
 
    # Set plot parameters
    ax.set_xticks(sigma)
    ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
    ax.set_ylabel('Number of iterations [-]',fontsize = 20)
    ax.set_title('Influence of $\\sigma$ on the number of iterations' + title_notes, fontweight ='bold', fontsize= 24, y= 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')

    if np.any(lim):
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        ax.set_xlim([-0.05, sigma[-1]+0.01])


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'nbr_iter_sig' + scenario, bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def APE_mu(type, data_vec, mu, description, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = [['red', 'green'], ['blue', 'purple']]

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                        markerfacecolor='k', markersize=10, linewidth=0)]

    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse_seq1 = np.zeros((3, len(mu)))
        ape_rmse_seq2 = np.zeros((3, len(mu)))
        for i in range(len(data)): # for every value of mu
            data_mu_i = data[i]

            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse_seq1[j,i] = np.mean(data_mu_i[j][::2]) # mean over all iterations for first sequence         
                ape_rmse_seq2[j,i] = np.mean(data_mu_i[j][1::2]) # mean over all iterations for second sequence   

        # Plot
        ax.plot(mu, ape_rmse_seq1[0], ':',   color = colors[k][0], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(mu, ape_rmse_seq1[1], 'x--', color = colors[k][0], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, ape_rmse_seq1[2], 'o--', color = colors[k][0], linewidth = 3, markersize = 10) # Data smoothed

        ax.plot(mu, ape_rmse_seq2[0], ':',   color = colors[k][1], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(mu, ape_rmse_seq2[1], 'x--', color = colors[k][1], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, ape_rmse_seq2[2], 'o--', color = colors[k][1], linewidth = 3, markersize = 10) # Data smoothed


    
    legend_elements.append(Line2D([0], [0], linestyle=':', color='k', label='Noisy', linewidth=3))
    
    for i in range(len(data_vec)):
        for j in range(2):
            legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i][j], label = labels[i][j], linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\mu$ [-]', fontsize = 20)
    ax.set_xticks(mu)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.0001, mu[-1] + 0.0001)
        elif description == 'unbounded':
             ax.set_xlim(-0.0001, mu[-1] + 0.0001)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in rotation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.0001, mu[-1] + 0.0001)
        elif description == 'unbounded':
             ax.set_xlim(-0.0001, mu[-1] + 0.0001)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')



    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'APE_rmse_{}_mu_{}'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()
    
def APE_mu_norm(type, data_vec, mu, description, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """

    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = [['red', 'green'], ['blue', 'purple']]

    ax.plot(np.linspace(mu[0]-0.0001, mu[-1]+0.0001, len(mu)),100*np.ones(len(mu)), 'k:', linewidth = 5)

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                        markerfacecolor='k', markersize=10, linewidth=0)]

    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse_seq1 = np.zeros((3, len(mu)))
        ape_rmse_seq2 = np.zeros((3, len(mu)))
        for i in range(len(data)): # for every value of mu
            data_mu_i = data[i]
            
            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse_seq1[j,i] = np.mean(data_mu_i[j][::2]) # mean over all iterations for first sequence         
                ape_rmse_seq2[j,i] = np.mean(data_mu_i[j][1::2]) # mean over all iterations for second sequence   
       
        # Normalize data
        data_BA_norm_seq1 = ape_rmse_seq1[1]/ape_rmse_seq1[0]*100
        data_smooth_norm_seq1 = ape_rmse_seq1[2]/ape_rmse_seq1[0]*100

        data_BA_norm_seq2 = ape_rmse_seq2[1]/ape_rmse_seq2[0]*100
        data_smooth_norm_seq2 = ape_rmse_seq2[2]/ape_rmse_seq2[0]*100
        
        # Plot
        ax.plot(mu, data_BA_norm_seq1, 'x--', color = colors[k][0], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, data_smooth_norm_seq1, 'o--', color = colors[k][0], linewidth = 3, markersize = 10) # Data smoothed

        ax.plot(mu, data_BA_norm_seq2, 'x--', color = colors[k][1], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, data_smooth_norm_seq2, 'o--', color = colors[k][1], linewidth = 3, markersize = 10) # Data smoothed
    
    for i in range(len(data_vec)):
        for j in range(2):
            legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i][j], label = labels[i][j], linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\mu$ [-]', fontsize = 20)
    ax.set_xticks(mu)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [%]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in translation: {}, {}'.format(description, 'normalized'),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'unbounded':
            ax.set_xlim(-0.0001, mu[-1]+0.0001)
        else:
            ax.set_xlim(-0.0009, mu[-1] + 0.0001)
            # ax.legend(handles = legend_elements, labelspacing = 1, fontsize = 15)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in rotation: {}, normalized'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.0001, mu[-1] + 0.0001)
        elif description == 'unbounded':
             ax.set_xlim(-0.0001, mu[-1] + 0.0001)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'APE_rmse_{}_mu_{}_norm'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def nbr_iterations_mu(data_vec, mu, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'blue']

    # Extract data
    for k, data in enumerate(data_vec):
        iterations_unlim = data[:,0,:]
        iterations_lim = data[:,-1,:]

        mean_iter_unlim, mean_iter_lim = [], []
        for i in range(iterations_unlim.shape[0]): # for every value of mu
            mean_iter_unlim.append(np.mean(iterations_unlim[i]))
            mean_iter_lim.append(np.mean(iterations_lim[i]))
        ax.plot(mu, mean_iter_unlim, 'x--', color = colors[k], linewidth = 3, markersize = 10) # Unbounded
        ax.plot(mu, mean_iter_lim, 'o--', color = colors[k], linewidth = 3, markersize = 10) # Bounded
    
    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='Unbounded',
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='Bounded',
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    for i in range(len(data_vec)):
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='{} & {}'.format(labels[i][0], labels[i][1]), linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1), handles = legend_elements, labelspacing = 1, fontsize = 15)
 
    # Set plot parameters
    ax.set_xticks(mu)
    ax.set_xlabel('$\\mu$ [-]', fontsize = 20)
    ax.set_ylabel('Number of iterations [-]',fontsize = 20)
    ax.set_title('Influence of $\\mu$ on the number of iterations',fontweight ='bold', fontsize= 24, y= 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
    ax.set_xlim([-0.0008, mu[-1]+0.0001])


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'nbr_iter_mu', bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def extract_time(timings, sigma):
    
    for i, time in enumerate(timings): # for every sequence
        time_tot_mean = []
        print("Sequence MH0"+str(i+1))
        for time_sig in time: # for every value of sigma
            for j in range(len(time_sig)):
                time_tot_mean.append(np.mean([time_vec[0] for time_vec in time_sig[j]]))

        print(len(time_tot_mean))
        # beautiful print
        sig = np.concatenate([np.repeat(x, 2) for x in sigma])
        for k, time in enumerate(time_tot_mean):
            if k % 2 == 0:
                print("Unbounded algorithm, sig =  {}, mean time = {}".format(sig[k], time))
            else:
                print("Bounded algorithm, sig =  {}, mean time = {}".format(sig[k], time))
                print(" ")

# Compare SABA to MABA
def APE_sigma_compare(type, data_vec_SABA, data_vec_MABA, sigma, description, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors = ['red', 'blue']

    # Set legend elements      
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA - SABA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='s', color='k', label='After BA - MABA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing - SABA',
                        markerfacecolor='k', markersize=10, linewidth=0),
                        Line2D([0], [0], marker='^', color='k', label='After smoothing - MABA',
                        markerfacecolor='k', markersize=10, linewidth=0), 
                        Line2D([0], [0], linestyle=':', color='k', label='Noisy - SABA', linewidth=3),
                        Line2D([0], [0], linestyle='-.', color='k', label='Noisy - MABA', linewidth=3)]
        
    for i in range(len(data_vec_MABA)):
            legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label = labels[i], linewidth=3))
    
    lgd = ax.legend(bbox_to_anchor=(0.30, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Extract data
    for k, (data_MABA, data_SABA) in enumerate(zip(data_vec_MABA, data_vec_SABA)):
        ape_rmse_seq1_MABA = np.zeros((3, len(sigma)))
        ape_rmse_seq1_SABA = np.zeros((3, len(sigma)))
        for i in range(len(data_MABA)): # for every value of sigma
            data_sig_i_MABA = data_MABA[i]
            data_sig_i_SABA = data_SABA[i]

            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse_seq1_MABA[j,i] = np.mean(data_sig_i_MABA[j][::2]) # mean over all iterations for first sequence
                ape_rmse_seq1_SABA[j, i] = np.mean(data_sig_i_SABA[j]) 

                # print(i, j, data_sig_i_MABA[j][::2])        
        # Plot
        ax.plot(sigma, ape_rmse_seq1_SABA[0], ':',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(sigma, ape_rmse_seq1_SABA[1], 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, ape_rmse_seq1_SABA[2], 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed

        ax.plot(sigma, ape_rmse_seq1_MABA[0], '-.',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(sigma, ape_rmse_seq1_MABA[1], 's--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, ape_rmse_seq1_MABA[2], '^--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed
    

    # Set plot parameters
    ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
    ax.set_xticks(sigma)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.07, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.01, sigma[-1] + 0.01)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in rotation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.07, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.05, sigma[-1] + 0.01)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'APE_rmse_{}_sig_{}'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def APE_mu_compare(type, data_vec_SABA, data_vec_MABA, mu, description, labels, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors = ['red', 'blue']

    # Set legend elements      
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA - SABA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='s', color='k', label='After BA - MABA',
                        markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing - SABA',
                        markerfacecolor='k', markersize=10, linewidth=0),
                        Line2D([0], [0], marker='^', color='k', label='After smoothing - MABA',
                        markerfacecolor='k', markersize=10, linewidth=0), 
                        Line2D([0], [0], linestyle=':', color='k', label='Noisy - SABA', linewidth=3),
                        Line2D([0], [0], linestyle='-.', color='k', label='Noisy - MABA', linewidth=3)]
        
    for i in range(len(data_vec_MABA)):
            legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label = labels[i], linewidth=3))
    
    lgd = ax.legend(bbox_to_anchor=(0.30, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Extract data
    for k, (data_MABA, data_SABA) in enumerate(zip(data_vec_MABA, data_vec_SABA)):
        ape_rmse_seq1_MABA = np.zeros((3, len(mu)))
        ape_rmse_seq1_SABA = np.zeros((3, len(mu)))
        for i in range(len(data_MABA)): # for every value of mu
            data_mu_i_MABA = data_MABA[i]
            data_mu_i_SABA = data_SABA[i]

            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse_seq1_MABA[j,i] = np.mean(data_mu_i_MABA[j][::2]) # mean over all iterations for first sequence
                ape_rmse_seq1_SABA[j, i] = np.mean(data_mu_i_SABA[j])         
        # Plot
        ax.plot(mu, ape_rmse_seq1_MABA[0], ':',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(mu, ape_rmse_seq1_MABA[1], 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, ape_rmse_seq1_MABA[2], 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed
    
        ax.plot(mu, ape_rmse_seq1_SABA[0], '-.',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(mu, ape_rmse_seq1_SABA[1], 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, ape_rmse_seq1_SABA[2], 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed


    # Set plot parameters
    ax.set_xlabel('$\\mu$ [-]', fontsize = 20)
    ax.set_xticks(mu)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.0001, mu[-1] + 0.0001)
        elif description == 'unbounded':
             ax.set_xlim(-0.0001, mu[-1] + 0.0001)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in rotation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.0001, mu[-1] + 0.0001)
        elif description == 'unbounded':
             ax.set_xlim(-0.0001, mu[-1] + 0.0001)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if save:
        plt.savefig(save_path + 'APE_rmse_{}_mu_{}'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

################################             Apply functions               ################################


sequences = ['MH01_MH02', 'MH04_MH05']


## Scenario 1: Influence of sig on Gaussian noise (camera position only)
sig_sc1 = np.linspace(0, 0.30, 11)
sc1_ape_trans_unlim, sc1_ape_trans_lim = [], []
sc1_ape_rot_unlim, sc1_ape_rot_lim = [], []
sc1_iter, sc1_timing = [], []

labels = [['MH01', 'MH02'], ['MH04', 'MH05']]

for seq in sequences:
    sc1_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc1_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc1_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc1_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/timings_{}.npy'.format(seq, seq)))

save = 0
save_path_sc1 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 1/'

# APE_sigma('translation', sc1_ape_trans_unlim, sig_sc1, 'unbounded', labels, save, save_path_sc1)
# APE_sigma('translation', sc1_ape_trans_lim, sig_sc1, 'bounded', labels, save, save_path_sc1)

# APE_sigma('rotation', sc1_ape_rot_unlim, sig_sc1, 'unbounded', labels, save, save_path_sc1)
# APE_sigma('rotation', sc1_ape_rot_lim, sig_sc1, 'bounded', labels, save, save_path_sc1)

# APE_sigma_norm('translation', sc1_ape_trans_unlim, sig_sc1, 'unbounded', labels, save, save_path_sc1)
# APE_sigma_norm('translation', sc1_ape_trans_lim, sig_sc1, 'bounded', labels, save, save_path_sc1)

# APE_sigma_norm('rotation', sc1_ape_rot_unlim, sig_sc1, 'unbounded', labels, save, save_path_sc1)
# APE_sigma_norm('rotation', sc1_ape_rot_lim, sig_sc1, 'bounded', labels, save, save_path_sc1)

# nbr_iterations_sigma(sc1_iter, sig_sc1, labels, save, save_path_sc1)
# extract_time(sc1_timing, sig_sc1)



## Scenario 1: Comparison of MABA and SABA results for sequence MH01 and MH04
sc1_ape_trans_unlim_SABA, sc1_ape_trans_lim_SABA = [], []
sc1_ape_rot_unlim_SABA, sc1_ape_rot_lim_SABA = [], []
sc1_iter_SABA = []

sc1_ape_trans_unlim_MABA, sc1_ape_trans_lim_MABA = [], []
sc1_ape_rot_unlim_MABA, sc1_ape_rot_lim_MABA = [], []
sc1_iter_MABA = []

labels = ['MH01', 'MH04']

for seq in labels:
    sc1_ape_trans_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_trans_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_rot_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc1_ape_rot_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc1_iter_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/nbr_iterations_{}.npy'.format(seq, seq)))

for seq in ['MH01_MH02', 'MH04_MH05']:
    sc1_ape_trans_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_trans_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_rot_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc1_ape_rot_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc1_iter_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 1/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc1_compare = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 1/Compare/'

# APE_sigma_compare('translation', sc1_ape_trans_unlim_SABA, sc1_ape_trans_unlim_MABA, sig_sc1, 'unbounded', labels, save, save_path_sc1_compare)
# APE_sigma_compare('translation', sc1_ape_trans_lim_SABA, sc1_ape_trans_lim_MABA, sig_sc1, 'bounded', labels, save, save_path_sc1_compare)

# APE_sigma_compare('rotation', sc1_ape_rot_unlim_SABA, sc1_ape_rot_unlim_MABA, sig_sc1, 'unbounded', labels, save, save_path_sc1_compare)
# APE_sigma_compare('rotation', sc1_ape_rot_lim_SABA, sc1_ape_rot_lim_MABA, sig_sc1, 'bounded', labels, save, save_path_sc1_compare)





## Scenario 2: Influence of sig on Gaussian noise (camera position + points)
sig_sc2 = np.linspace(0, 0.15, 11)
sc2_ape_trans_unlim, sc2_ape_trans_lim = [], []
sc2_ape_rot_unlim, sc2_ape_rot_lim = [], []
sc2_iter, sc2_timing = [], []
labels = [['MH01', 'MH02'], ['MH04', 'MH05']]

for seq in sequences:
    sc2_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc2_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc2_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc2_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/timings_{}.npy'.format(seq, seq)))


save = 0
save_path_sc2 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 2/'

# APE_sigma('translation', sc2_ape_trans_unlim, sig_sc2, 'unbounded', labels, save, save_path_sc2)
# APE_sigma('translation', sc2_ape_trans_lim, sig_sc2, 'bounded', labels, save, save_path_sc2)

# APE_sigma('rotation', sc2_ape_rot_unlim, sig_sc2, 'unbounded', labels, save, save_path_sc2)
# APE_sigma('rotation', sc2_ape_rot_lim, sig_sc2, 'bounded', labels, save, save_path_sc2)

# APE_sigma_norm('translation', sc2_ape_trans_unlim, sig_sc2, 'unbounded', labels, save, save_path_sc2)
# APE_sigma_norm('translation', sc2_ape_trans_lim, sig_sc2, 'bounded', labels, save, save_path_sc2)

# APE_sigma_norm('rotation', sc2_ape_rot_unlim, sig_sc2, 'unbounded', labels, save, save_path_sc2)
# APE_sigma_norm('rotation', sc2_ape_rot_lim, sig_sc2, 'bounded', labels, save, save_path_sc2)
# extract_time(sc2_timing, sig_sc2)

## Scenario 2: Comparison of MABA and SABA results for sequence MH01 and MH04
sc2_ape_trans_unlim_SABA, sc2_ape_trans_lim_SABA = [], []
sc2_ape_rot_unlim_SABA, sc2_ape_rot_lim_SABA = [], []
sc2_iter_SABA = []

sc2_ape_trans_unlim_MABA, sc2_ape_trans_lim_MABA = [], []
sc2_ape_rot_unlim_MABA, sc2_ape_rot_lim_MABA = [], []
sc2_iter_MABA = []

labels = ['MH01', 'MH04']

for seq in labels:
    sc2_ape_trans_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_trans_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_rot_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc2_ape_rot_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc2_iter_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/nbr_iterations_{}.npy'.format(seq, seq)))

for seq in ['MH01_MH02', 'MH04_MH05']:
    sc2_ape_trans_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_trans_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_rot_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc2_ape_rot_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc2_iter_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 2/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc2_compare = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 2/Compare/'

# APE_sigma_compare('translation', sc2_ape_trans_unlim_SABA, sc2_ape_trans_unlim_MABA, sig_sc2, 'unbounded', labels, save, save_path_sc2_compare)
# APE_sigma_compare('translation', sc2_ape_trans_lim_SABA, sc2_ape_trans_lim_MABA, sig_sc2, 'bounded', labels, save, save_path_sc2_compare)

# APE_sigma_compare('rotation', sc2_ape_rot_unlim_SABA, sc2_ape_rot_unlim_MABA, sig_sc2, 'unbounded', labels, save, save_path_sc2_compare)
# APE_sigma_compare('rotation', sc2_ape_rot_lim_SABA, sc2_ape_rot_lim_MABA, sig_sc2, 'bounded', labels, save, save_path_sc2_compare)



# ## Scenario 3: Influence of sig on Gaussian noise (camera position + points)
sig_sc3 = np.linspace(0, 0.15, 11)
sc3_ape_trans_unlim, sc3_ape_trans_lim = [], []
sc3_ape_rot_unlim, sc3_ape_rot_lim = [], []
sc3_iter, sc3_timing = [], []
labels = [['MH01', 'MH02'], ['MH04', 'MH05']]
# sequences = ['MH01_MH02']
for seq in sequences:
    sc3_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc3_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc3_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc3_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/timings_{}.npy'.format(seq, seq)))

save = 0
save_path_sc3 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 3/'


# APE_sigma('translation', sc3_ape_trans_unlim, sig_sc3, 'unbounded', labels, save, save_path_sc3)
# APE_sigma('translation', sc3_ape_trans_lim, sig_sc3, 'bounded', labels, save, save_path_sc3)

# APE_sigma('rotation', sc3_ape_rot_unlim, sig_sc3, 'unbounded', labels, save, save_path_sc3)
# APE_sigma('rotation', sc3_ape_rot_lim, sig_sc3, 'bounded', labels, save, save_path_sc3)

# APE_sigma_norm('translation', sc3_ape_trans_unlim, sig_sc3, 'unbounded', labels, save, save_path_sc3)
# APE_sigma_norm('translation', sc3_ape_trans_lim, sig_sc3, 'bounded', labels, save, save_path_sc3)

# APE_sigma_norm('rotation', sc3_ape_rot_unlim, sig_sc3, 'unbounded', labels, save, save_path_sc3)
# APE_sigma_norm('rotation', sc3_ape_rot_lim, sig_sc3, 'bounded', labels, save, save_path_sc3)

# nbr_iterations_sigma(sc3_iter, sig_sc3, labels, save, save_path_sc3)
# extract_time(sc3_timing, sig_sc3)

## Scenario 3: Comparison of MABA and SABA results for sequence MH01 and MH04
sc3_ape_trans_unlim_SABA, sc3_ape_trans_lim_SABA = [], []
sc3_ape_rot_unlim_SABA, sc3_ape_rot_lim_SABA = [], []
sc3_iter_SABA = []

sc3_ape_trans_unlim_MABA, sc3_ape_trans_lim_MABA = [], []
sc3_ape_rot_unlim_MABA, sc3_ape_rot_lim_MABA = [], []
sc3_iter_MABA = []

labels = ['MH01', 'MH04']

for seq in labels:
    sc3_ape_trans_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_trans_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_rot_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc3_ape_rot_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc3_iter_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/nbr_iterations_{}.npy'.format(seq, seq)))

for seq in ['MH01_MH02', 'MH04_MH05']:
    sc3_ape_trans_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_trans_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_rot_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc3_ape_rot_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc3_iter_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 3/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc3_compare = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 3/Compare/'

# APE_sigma_compare('translation', sc3_ape_trans_unlim_SABA, sc3_ape_trans_unlim_MABA, sig_sc3, 'unbounded', labels, save, save_path_sc3_compare)
# APE_sigma_compare('translation', sc3_ape_trans_lim_SABA, sc3_ape_trans_lim_MABA, sig_sc3, 'bounded', labels, save, save_path_sc3_compare)

# APE_sigma_compare('rotation', sc3_ape_rot_unlim_SABA, sc3_ape_rot_unlim_MABA, sig_sc3, 'unbounded', labels, save, save_path_sc3_compare)
# APE_sigma_compare('rotation', sc3_ape_rot_lim_SABA, sc3_ape_rot_lim_MABA, sig_sc3, 'bounded', labels, save, save_path_sc3_compare)

save_path_sc2sc3 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Comparison/Sc2_Sc3/'
# nbr_iterations_sigma(sc2_iter, sig_sc2, save, save_path_sc2sc3, '_scenario_2', ': Scenario 2', lim = [-0.05, 0.16, 0, 275])
# nbr_iterations_sigma(sc3_iter, sig_sc3, save, save_path_sc2sc3, '_scenario_3', ': Scenario 3', lim = [-0.05, 0.16, 0, 275])


## Scenario 4: Influence of mu on Random Walk noise (camera position only)
mu_sc4 = np.linspace(0, 0.0025, 11)
sc4_ape_trans_unlim, sc4_ape_trans_lim = [], []
sc4_ape_rot_unlim, sc4_ape_rot_lim = [], []
sc4_iter = []

labels = [['MH01', 'MH02'], ['MH04', 'MH05']]

for seq in sequences:
    sc4_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc4_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc4_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc4 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 4/'

# APE_mu('translation', sc4_ape_trans_unlim, mu_sc4, 'unbounded', labels, save, save_path_sc4)
# APE_mu('translation', sc4_ape_trans_lim, mu_sc4, 'bounded', labels, save, save_path_sc4)

# APE_mu('rotation', sc4_ape_rot_unlim, mu_sc4, 'unbounded', labels, save, save_path_sc4)
# APE_mu('rotation', sc4_ape_rot_lim, mu_sc4, 'bounded', labels, save, save_path_sc4)

# APE_mu_norm('translation', sc4_ape_trans_unlim, mu_sc4, 'unbounded', labels, save, save_path_sc4)
# APE_mu_norm('translation', sc4_ape_trans_lim, mu_sc4, 'bounded', labels, save, save_path_sc4)

# APE_mu_norm('rotation', sc4_ape_rot_unlim, mu_sc4, 'unbounded', labels, save, save_path_sc4)
# APE_mu_norm('rotation', sc4_ape_rot_lim, mu_sc4, 'bounded', labels, save, save_path_sc4)

# nbr_iterations_mu(sc4_iter, mu_sc4, labels, save, save_path_sc4)

## Scenario 4: Comparison of MABA and SABA results for sequence MH01 and MH04
sc4_ape_trans_unlim_SABA, sc4_ape_trans_lim_SABA = [], []
sc4_ape_rot_unlim_SABA, sc4_ape_rot_lim_SABA = [], []
sc4_iter_SABA = []

sc4_ape_trans_unlim_MABA, sc4_ape_trans_lim_MABA = [], []
sc4_ape_rot_unlim_MABA, sc4_ape_rot_lim_MABA = [], []
sc4_iter_MABA = []

labels = ['MH01', 'MH04']

for seq in labels:
    sc4_ape_trans_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_trans_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_rot_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc4_ape_rot_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc4_iter_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/nbr_iterations_{}.npy'.format(seq, seq)))

for seq in ['MH01_MH02', 'MH04_MH05']:
    sc4_ape_trans_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_trans_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_rot_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc4_ape_rot_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc4_iter_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 4/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc4_compare = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 4/Compare/'

# APE_mu_compare('translation', sc4_ape_trans_unlim_SABA, sc4_ape_trans_unlim_MABA, mu_sc4, 'unbounded', labels, save, save_path_sc4_compare)
# APE_mu_compare('translation', sc4_ape_trans_lim_SABA, sc4_ape_trans_lim_MABA, mu_sc4, 'bounded', labels, save, save_path_sc4_compare)

# APE_mu_compare('rotation', sc4_ape_rot_unlim_SABA, sc4_ape_rot_unlim_MABA, mu_sc4, 'unbounded', labels, save, save_path_sc4_compare)
# APE_mu_compare('rotation', sc4_ape_rot_lim_SABA, sc4_ape_rot_lim_MABA, mu_sc4, 'bounded', labels, save, save_path_sc4_compare)




## Scenario 5: Influence of sigma on Random Walk noise applied to points
sig_sc5 = np.linspace(0, 0.15, 11)
sc5_ape_trans_unlim, sc5_ape_trans_lim = [], []
sc5_ape_rot_unlim, sc5_ape_rot_lim = [], []
sc5_iter, sc5_timing = [], []

labels = [['MH01', 'MH02'], ['MH04', 'MH05']]
sequences = ['MH01_MH02', 'MH04_MH05']
for seq in sequences:
    sc5_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc5_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc5_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc5_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/timings_{}.npy'.format(seq, seq)))


save = 0
save_path_sc5 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 5/'

# APE_sigma('translation', sc5_ape_trans_unlim, sig_sc5, 'unbounded', labels, save, save_path_sc5)
# APE_sigma('translation', sc5_ape_trans_lim, sig_sc5, 'bounded', labels, save, save_path_sc5)

# APE_sigma('rotation', sc5_ape_rot_unlim, sig_sc5, 'unbounded', labels, save, save_path_sc5)
# APE_sigma('rotation', sc5_ape_rot_lim, sig_sc5, 'bounded', labels, save, save_path_sc5)

# APE_sigma_norm('translation', sc5_ape_trans_unlim, sig_sc5, 'unbounded', labels, save, save_path_sc5)
# APE_sigma_norm('translation', sc5_ape_trans_lim, sig_sc5, 'bounded', labels, save, save_path_sc5)

# APE_sigma_norm('rotation', sc5_ape_rot_unlim, sig_sc5, 'unbounded', labels, save, save_path_sc5)
# APE_sigma_norm('rotation', sc5_ape_rot_lim, sig_sc5, 'bounded', labels, save, save_path_sc5)

# nbr_iterations_sigma(sc5_iter, sig_sc5, labels, save, save_path_sc5)
# extract_time(sc5_timing, sig_sc5)

## Scenario 5: Comparison of MABA and SABA results for sequence MH01 and MH04
sc5_ape_trans_unlim_SABA, sc5_ape_trans_lim_SABA = [], []
sc5_ape_rot_unlim_SABA, sc5_ape_rot_lim_SABA = [], []
sc5_iter_SABA = []

sc5_ape_trans_unlim_MABA, sc5_ape_trans_lim_MABA = [], []
sc5_ape_rot_unlim_MABA, sc5_ape_rot_lim_MABA = [], []
sc5_iter_MABA = []

labels = ['MH01', 'MH04']

for seq in labels:
    sc5_ape_trans_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_trans_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_rot_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc5_ape_rot_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc5_iter_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/nbr_iterations_{}.npy'.format(seq, seq)))

for seq in ['MH01_MH02', 'MH04_MH05']:
    sc5_ape_trans_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_trans_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_rot_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc5_ape_rot_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc5_iter_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 5/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc5_compare = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 5/Compare/'

# APE_sigma_compare('translation', sc5_ape_trans_unlim_SABA, sc5_ape_trans_unlim_MABA, sig_sc5, 'unbounded', labels, save, save_path_sc5_compare)
# APE_sigma_compare('translation', sc5_ape_trans_lim_SABA, sc5_ape_trans_lim_MABA, sig_sc5, 'bounded', labels, save, save_path_sc5_compare)

# APE_sigma_compare('rotation', sc5_ape_rot_unlim_SABA, sc5_ape_rot_unlim_MABA, sig_sc5, 'unbounded', labels, save, save_path_sc5_compare)
# APE_sigma_compare('rotation', sc5_ape_rot_lim_SABA, sc5_ape_rot_lim_MABA, sig_sc5, 'bounded', labels, save, save_path_sc5_compare)

# ## Scenario 6: Influence of sig on Gaussian noise (camera position + points)
sig_sc6 = np.linspace(0, 0.15, 11)
sc6_ape_trans_unlim, sc6_ape_trans_lim = [], []
sc6_ape_rot_unlim, sc6_ape_rot_lim = [], []
sc6_iter, sc6_timing = [], []
labels = [['MH01', 'MH02'], ['MH04', 'MH05']]
# sequences = ['MH01_MH02']
for seq in sequences:
    sc6_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc6_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc6_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc6_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/timings_{}.npy'.format(seq, seq)))

save = 0
save_path_sc6 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 6/'


# APE_sigma('translation', sc6_ape_trans_unlim, sig_sc6, 'unbounded', labels, save, save_path_sc6)
# APE_sigma('translation', sc6_ape_trans_lim, sig_sc6, 'bounded', labels, save, save_path_sc6)

# APE_sigma('rotation', sc6_ape_rot_unlim, sig_sc6, 'unbounded', labels, save, save_path_sc6)
# APE_sigma('rotation', sc6_ape_rot_lim, sig_sc6, 'bounded', labels, save, save_path_sc6)

# APE_sigma_norm('translation', sc6_ape_trans_unlim, sig_sc6, 'unbounded', labels, save, save_path_sc6)
# APE_sigma_norm('translation', sc6_ape_trans_lim, sig_sc6, 'bounded', labels, save, save_path_sc6)

# APE_sigma_norm('rotation', sc6_ape_rot_unlim, sig_sc6, 'unbounded', labels, save, save_path_sc6)
# APE_sigma_norm('rotation', sc6_ape_rot_lim, sig_sc6, 'bounded', labels, save, save_path_sc6)

# nbr_iterations_sigma(sc6_iter, sig_sc6, labels, save, save_path_sc6)
extract_time(sc6_timing, sig_sc6)


## Scenario 6: Comparison of MABA and SABA results for sequence MH01 and MH04
sc6_ape_trans_unlim_SABA, sc6_ape_trans_lim_SABA = [], []
sc6_ape_rot_unlim_SABA, sc6_ape_rot_lim_SABA = [], []
sc6_iter_SABA = []

sc6_ape_trans_unlim_MABA, sc6_ape_trans_lim_MABA = [], []
sc6_ape_rot_unlim_MABA, sc6_ape_rot_lim_MABA = [], []
sc6_iter_MABA = []

labels = ['MH01', 'MH04']

for seq in labels:
    sc6_ape_trans_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_trans_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_rot_unlim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc6_ape_rot_lim_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc6_iter_SABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/nbr_iterations_{}.npy'.format(seq, seq)))

for seq in ['MH01_MH02', 'MH04_MH05']:
    sc6_ape_trans_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_trans_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_rot_unlim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc6_ape_rot_lim_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc6_iter_MABA.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/nbr_iterations_{}.npy'.format(seq, seq)))


save = 0
save_path_sc6_compare = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Scenario 6/Compare/'

# APE_sigma_compare('translation', sc6_ape_trans_unlim_SABA, sc6_ape_trans_unlim_MABA, sig_sc6, 'unbounded', labels, save, save_path_sc6_compare)
# APE_sigma_compare('translation', sc6_ape_trans_lim_SABA, sc6_ape_trans_lim_MABA, sig_sc6, 'bounded', labels, save, save_path_sc6_compare)

# APE_sigma_compare('rotation', sc6_ape_rot_unlim_SABA, sc6_ape_rot_unlim_MABA, sig_sc6, 'unbounded', labels, save, save_path_sc6_compare)
# APE_sigma_compare('rotation', sc6_ape_rot_lim_SABA, sc6_ape_rot_lim_MABA, sig_sc6, 'bounded', labels, save, save_path_sc6_compare)

save_path_sc5sc6 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/MABA/Comparison/Sc5_Sc6/'
# nbr_iterations_sigma(sc5_iter, sig_sc5, save, save_path_sc5sc6, '_scenario_5', ': Scenario 5', lim = [-0.02, 0.16, 0, 250])
# nbr_iterations_sigma(sc6_iter, sig_sc6, save, save_path_sc5sc6, '_scenario_6', ': Scenario 6', lim = [-0.02, 0.16, 0, 250])