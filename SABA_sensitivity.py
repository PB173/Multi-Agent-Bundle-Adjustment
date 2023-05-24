##################### Import modules and functions #####################
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

################################             Create functions               ################################

# def sensitivity_plot_sequential(input_data, sigma, nbr_iter, description):
#     """

#     Arguments:

#     Returns:
  
#     """
#     # Initialize figures
#     fig, ax = plt.subplots(1,1, figsize = (16,8))
#     fig.subplots_adjust(left=0.16)

#     # Extract data
#     for k in range(len(input_data)):
#         data = input_data[k]
#         print(data)
#         for i in range(len(data)):
#             ape_rmse = []
#             for j in range(len(sigma)):
#                 print("considered data:", data[i][j*nbr_iter:(j+1)*nbr_iter])
#                 ape_rmse.append(np.mean(data[i][j*nbr_iter:(j+1)*nbr_iter]))
                
#             # Plot
#             ax.plot(sigma, ape_rmse, 'x--')

#     # Set plot parameters
#     labels= ('Noisy data', 'After BA', 'After BA and smoothing')
#     ax.legend(labels, fontsize = 24, markerscale=2)

#     ax.set_xlabel('$\\sigma$', fontsize = 20)
#     ax.set_ylabel('RMSE APE translation',fontsize = 20)
#     ax.set_title('Influence of $\\sigma$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
#     ax.minorticks_on()
#     ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
#     ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')

#     # if save:
#     #     plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Cost Functions/{}/Cost_MH0{}.svg".format(ORB_version, seq_idx), 
#     #             bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
#     # else:
#     #     plt.show()
#     plt.show()

# def sensitivity_plot_parallel(data, sigma, nbr_iter, description):
#     """

#     Arguments:

#     Returns:
  
#     """
#     # Initialize figures
#     fig, ax = plt.subplots(1,1, figsize = (16,8))
#     fig.subplots_adjust(left=0.16)

#     # Extract data
#     ape_rmse = np.zeros((3, len(sigma)))
#     for i in range(len(data)): # for every value of sigma
#         data_sig_i = data[i]
#         print("data sig = {}:\n".format(sigma[i]), data_sig_i)
#         for j in range(3): # for every type: noisy, BA and smooth
#             print("considered data:",data_sig_i[j])
#             print("mean considered data:", np.mean(data_sig_i[j]))
#             ape_rmse[j,i] = np.mean(data_sig_i[j]) # mean over all iterations
#         print("APE RMSE:\n", ape_rmse)            
#     # Plot
#     for i in range(3):
#         ax.plot(sigma, ape_rmse[i], 'x--', linewidth = 3, markersize = 10)

#     # Set plot parameters
#     labels= ('Noisy data', 'After BA', 'After BA and smoothing')
#     ax.legend(labels, fontsize = 24, markerscale=2)

#     ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
#     ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
#     ax.set_title('Influence of $\\sigma$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.minorticks_on()
#     ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
#     ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')

#     # if save:
#     #     plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Cost Functions/{}/Cost_MH0{}.svg".format(ORB_version, seq_idx), 
#     #             bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
#     # else:
#     #     plt.show()
#     plt.show()

# def sensitiivity_iteration(iterations, sigma):
#     # Initialize figures
#     fig, ax = plt.subplots(1,1, figsize = (16,8))
#     fig.subplots_adjust(left=0.16)

#     # Extract data
#     iterations_unlim = iterations[:,0,:]
#     iterations_lim = iterations[:,1,:]
#     for i in range(iterations_unlim.shape[0]):
#         ax.plot(iterations_unlim[i,:],'x--', label='Unbounded, $\\sigma$ = {}'.format(sigma[i]))
#     for i in range(iterations_lim.shape[0]):
#         ax.plot(iterations_lim[i,:], 'o--', label='Bounded, $\\sigma$ = {}'.format(sigma[i]))
    
#     # Set plot parameters
#     ax.legend(fontsize = 24, markerscale=2)

#     ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
#     ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
#     ax.set_title('Influence of $\\sigma$ on the number of iterations',fontweight ='bold', fontsize= 24, y= 1)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.minorticks_on()
#     ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
#     ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')

#     plt.show()


def APE_sigma(type, data_vec, sigma, description, save=0, save_path = '', lim = []):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'green', 'orange', 'blue', 'purple']

    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse = np.zeros((3, len(sigma)))
        for i in range(len(data)): # for every value of sigma
            data_sig_i = data[i]
            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse[j,i] = np.mean(data_sig_i[j]) # mean over all iterations         
       
        # Plot
        ax.plot(sigma, ape_rmse[0], ':',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(sigma, ape_rmse[1], 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, ape_rmse[2], 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    legend_elements.append(Line2D([0], [0], linestyle=':', color='k', label='Noisy', linewidth=3))
    for i in range(len(data_vec)):
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
    ax.set_xticks(sigma)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.01, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.01, sigma[-1] + 0.01)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in rotation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.05, sigma[-1] + 0.01)
        elif description == 'unbounded':
             ax.set_xlim(-0.05, sigma[-1] + 0.01)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0)), ax.xaxis.get_offset_text().set_size(18)

    if np.any(lim):
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])


    if save:
        plt.savefig(save_path + 'APE_rmse_{}_sig_{}'.format(type, description), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def APE_sigma_norm(type, data_vec, sigma, description, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'green', 'orange', 'blue', 'purple']

    ax.plot(np.linspace(sigma[0]-0.01, sigma[-1]+0.02, len(sigma)),100*np.ones(len(sigma)), 'k:', linewidth = 5)
    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse = np.zeros((3, len(sigma)))
        for i in range(len(data)): # for every value of sigma
            data_sig_i = data[i]
            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse[j,i] = np.mean(data_sig_i[j]) # mean over all iterations         
        # Normalize data
        data_BA_norm = ape_rmse[1]/ape_rmse[0]*100
        data_smooth_norm = ape_rmse[2]/ape_rmse[0]*100
        # Plot
        # ax.plot(sigma, ape_rmse[0], ':',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(sigma,data_BA_norm, 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(sigma, data_smooth_norm, 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    # legend_elements.append(Line2D([0], [0], linestyle=':', color='k', label='Noisy', linewidth=3))
    for i in range(len(data_vec)):
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
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
            ax.set_xlim(-0.01, sigma[-1] + 0.01)
            # ax.legend(handles = legend_elements, labelspacing = 1, fontsize = 15)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [%]',fontsize = 20)
        ax.set_title('Influence of $\\sigma$ on the APE in rotation: {}, {}'.format(description, 'normalized'),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'unbounded':
            ax.set_xlim(-0.01, sigma[-1]+0.01)
        if description == 'bounded':
            
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.50), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.01, sigma[-1]+0.01)
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
    
    for i in range(len(data_vec)):
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
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

def APE_mu(type, data_vec, mu, description, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'green', 'orange', 'blue', 'purple']

    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse = np.zeros((3, len(mu)))
        for i in range(len(data)): # for every value of sigma
            data_mu_i = data[i]
            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse[j,i] = np.mean(data_mu_i[j]) # mean over all iterations  
                # print("Iterations for i = {} and j = {}: \n {}".format(i, j, data_mu_i[j]))
                       
       
        # Plot
        ax.plot(mu, ape_rmse[0], ':',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(mu, ape_rmse[1], 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, ape_rmse[2], 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    legend_elements.append(Line2D([0], [0], linestyle=':', color='k', label='Noisy', linewidth=3))
    for i in range(len(data_vec)):
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\mu$ [-]', fontsize = 20)
    ax.set_xticks(mu)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [m]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in translation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [°]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in rotation: {}'.format(description),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'bounded':
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.53), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(mu[0]-0.0008, mu[-1] + 0.0001)
        elif description == 'unbounded':
             ax.set_xlim(mu[0]-0.0008, mu[-1] + 0.0001)
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

def APE_mu_norm(type, data_vec, mu, description, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'green', 'orange', 'blue', 'purple']

    ax.plot(np.linspace(mu[0]-0.0001, mu[-1]+0.0001, len(mu)),100*np.ones(len(mu)), 'k:', linewidth = 5)
    # Extract data
    for k, data in enumerate(data_vec):
        ape_rmse = np.zeros((3, len(mu)))
        for i in range(len(data)): # for every value of sigma
            data_sig_i = data[i]
            for j in range(3): # for every type: noisy, BA and smooth
                ape_rmse[j,i] = np.mean(data_sig_i[j]) # mean over all iterations         
        # Normalize data
        data_BA_norm = ape_rmse[1]/ape_rmse[0]*100
        data_smooth_norm = ape_rmse[2]/ape_rmse[0]*100
        # Plot
        # ax.plot(sigma, ape_rmse[0], ':',   color = colors[k], linewidth = 3, markersize = 10) # Noisy data
        ax.plot(mu,data_BA_norm, 'x--', color = colors[k], linewidth = 3, markersize = 10) # Data after BA
        ax.plot(mu, data_smooth_norm, 'o--', color = colors[k], linewidth = 3, markersize = 10) # Data smoothed

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label='After BA',
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label='After smoothing',
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    # legend_elements.append(Line2D([0], [0], linestyle=':', color='k', label='Noisy', linewidth=3))
    for i in range(len(data_vec)):
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.22, 1.0), handles = legend_elements, labelspacing = 1, fontsize = 15)

    # Set plot parameters
    ax.set_xlabel('$\\mu$ [-]', fontsize = 20)
    ax.set_xticks(mu)
    if type == 'translation':
        ax.set_ylabel('RMSE APE translation [%]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in translation: {}, {}'.format(description, 'normalized'),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'unbounded':
            ax.set_xlim(-0.0009, mu[-1]+0.0001)
        else:
            ax.set_xlim(-0.0009, mu[-1] + 0.0001)
    elif type == 'rotation':
        ax.set_ylabel('RMSE APE rotation [%]',fontsize = 20)
        ax.set_title('Influence of $\\mu$ on the APE in rotation: {}, {}'.format(description, 'normalized'),fontweight ='bold', fontsize= 24, y= 1)
        if description == 'unbounded':
            ax.set_xlim(-0.0001, mu[-1]+0.0001)
        if description == 'bounded':
            
            # lgd = ax.legend(bbox_to_anchor=(0.22, 0.50), handles = legend_elements, labelspacing = 1, fontsize = 15)
            ax.set_xlim(-0.0001, mu[-1]+0.0001)
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

def nbr_iterations_mu(data_vec, mu, save=0, save_path = ''):
    """

    Arguments:

    Returns:
  
    """
    # Initialize figures
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    fig.subplots_adjust(left=0.16)
    colors  = ['red', 'green', 'orange', 'blue', 'purple']

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
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label='MH0{}'.format(i+1), linewidth=3))
    lgd = ax.legend(bbox_to_anchor=(0.18, 1), handles = legend_elements, labelspacing = 1, fontsize = 15)
 
    # Set plot parameters
    ax.set_xticks(mu)
    ax.set_xlabel('$\\sigma$ [-]', fontsize = 20)
    ax.set_ylabel('Number of iterations [-]',fontsize = 20)
    ax.set_title('Influence of $\\mu$ on the number of iterations',fontweight ='bold', fontsize= 24, y= 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
    ax.set_xlim([mu[0]-0.0002, mu[-1]+0.0002])

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
################################             Apply functions               ################################


sequences = ['MH01', 'MH02', 'MH03', 'MH04', 'MH05']

## Scenario 1: Influence of sig on Gaussian noise (camera position only)
sig_sc1 = np.linspace(0, 0.30, 11)
sc1_ape_trans_unlim, sc1_ape_trans_lim = [], []
sc1_ape_rot_unlim, sc1_ape_rot_lim = [], []
sc1_iter, sc1_timing = [], []

for seq in sequences:
    sc1_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc1_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc1_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc1_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc1_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 1/{}/timings_{}.npy'.format(seq, seq)))



save = 0
save_path_sc1 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 1/'

# APE_sigma('translation', sc1_ape_trans_unlim, sig_sc1, 'unbounded', save, save_path_sc1)
# APE_sigma('translation', sc1_ape_trans_lim, sig_sc1, 'bounded', save, save_path_sc1)

# APE_sigma('rotation', sc1_ape_rot_unlim, sig_sc1, 'unbounded', save, save_path_sc1)
# APE_sigma('rotation', sc1_ape_rot_lim, sig_sc1, 'bounded', save, save_path_sc1)

# APE_sigma_norm('translation', sc1_ape_trans_unlim, sig_sc1, 'unbounded', save, save_path_sc1)
# APE_sigma_norm('translation', sc1_ape_trans_lim, sig_sc1, 'bounded', save, save_path_sc1)

# APE_sigma_norm('rotation', sc1_ape_rot_unlim, sig_sc1, 'unbounded', save, save_path_sc1)
# APE_sigma_norm('rotation', sc1_ape_rot_lim, sig_sc1, 'bounded', save, save_path_sc1)


# nbr_iterations_sigma(sc1_iter, sig_sc1, save, save_path_sc1)
# extract_time(sc1_timing, sig_sc1)


## Scenario 2: Influence of sig on Gaussian noise (camera position + points)
sig_sc2 = np.linspace(0, 0.15, 11)
sc2_ape_trans_unlim, sc2_ape_trans_lim = [], []
sc2_ape_rot_unlim, sc2_ape_rot_lim = [], []
sc2_iter, sc2_timing = [], []
for seq in sequences:
    sc2_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc2_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc2_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc2_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc2_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 2/{}/timings_{}.npy'.format(seq, seq)))

save = 0
save_path_sc2 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 2/'

# APE_sigma('translation', sc2_ape_trans_unlim, sig_sc2, 'unbounded', save, save_path_sc2)
# APE_sigma('translation', sc2_ape_trans_lim, sig_sc2, 'bounded', save, save_path_sc2)

# APE_sigma('rotation', sc2_ape_rot_unlim, sig_sc2, 'unbounded', save, save_path_sc2)
# APE_sigma('rotation', sc2_ape_rot_lim, sig_sc2, 'bounded', save, save_path_sc2)

# APE_sigma_norm('translation', sc2_ape_trans_unlim, sig_sc2, 'unbounded', save, save_path_sc2)
# APE_sigma_norm('translation', sc2_ape_trans_lim, sig_sc2, 'bounded', save, save_path_sc2)

# APE_sigma_norm('rotation', sc2_ape_rot_unlim, sig_sc2, 'unbounded', save, save_path_sc2)
# APE_sigma_norm('rotation', sc2_ape_rot_lim, sig_sc2, 'bounded', save, save_path_sc2)


# nbr_iterations_sigma(sc2_iter, sig_sc2, save, save_path_sc2)
# extract_time(sc2_timing, sig_sc2)

## Scenario 3: Influence of sig on Gaussian noise (camera position + points) => variation noise on points
sig_sc3 = np.linspace(0, 0.15, 11)
sc3_ape_trans_unlim, sc3_ape_trans_lim = [], []
sc3_ape_rot_unlim, sc3_ape_rot_lim = [], []
sc3_iter, sc3_timing = [], []
for seq in sequences:
    sc3_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc3_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc3_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc3_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc3_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3/{}/timings_{}.npy'.format(seq, seq)))

save = 0
save_path_sc3 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 3/'


# APE_sigma('translation', sc3_ape_trans_unlim, sig_sc3, 'unbounded', save, save_path_sc3)
# APE_sigma('translation', sc3_ape_trans_lim, sig_sc3, 'bounded', save, save_path_sc3)

# APE_sigma('rotation', sc3_ape_rot_unlim, sig_sc3, 'unbounded', save, save_path_sc3)
# APE_sigma('rotation', sc3_ape_rot_lim, sig_sc3, 'bounded', save, save_path_sc3)

# APE_sigma_norm('translation', sc3_ape_trans_unlim, sig_sc3, 'unbounded', save, save_path_sc3)
# APE_sigma_norm('translation', sc3_ape_trans_lim, sig_sc3, 'bounded', save, save_path_sc3)

# APE_sigma_norm('rotation', sc3_ape_rot_unlim, sig_sc3, 'unbounded', save, save_path_sc3)
# APE_sigma_norm('rotation', sc3_ape_rot_lim, sig_sc3, 'bounded', save, save_path_sc3)

# nbr_iterations_sigma(sc3_iter, sig_sc3, save, save_path_sc3)
# extract_time(sc3_timing, sig_sc3)

## Scenario 3 cam: Influence of sig on Gaussian noise (camera position + points) => variation noise on camera
sig_sc3_cam = np.linspace(0, 0.30, 11)
sc3_cam_ape_trans_unlim, sc3_cam_ape_trans_lim = [], []
sc3_cam_ape_rot_unlim, sc3_cam_ape_rot_lim = [], []
sc3_cam_iter = []
for seq in sequences:
    sc3_cam_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3_cam/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc3_cam_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3_cam/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc3_cam_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3_cam/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc3_cam_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3_cam/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc3_cam_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3_cam/{}/nbr_iterations_{}.npy'.format(seq, seq)))

save = 0
save_path_sc3_cam = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 3_cam/'


# APE_sigma('translation', sc3_cam_ape_trans_unlim, sig_sc3_cam, 'unbounded', save, save_path_sc3_cam)
# APE_sigma('translation', sc3_cam_ape_trans_lim, sig_sc3_cam, 'bounded', save, save_path_sc3_cam)

# APE_sigma('rotation', sc3_cam_ape_rot_unlim, sig_sc3_cam, 'unbounded', save, save_path_sc3_cam)
# APE_sigma('rotation', sc3_cam_ape_rot_lim, sig_sc3_cam, 'bounded', save, save_path_sc3_cam)

# APE_sigma_norm('translation', sc3_cam_ape_trans_unlim, sig_sc3_cam, 'unbounded', save, save_path_sc3_cam)
# APE_sigma_norm('translation', sc3_cam_ape_trans_lim, sig_sc3_cam, 'bounded', save, save_path_sc3_cam)

# APE_sigma_norm('rotation', sc3_cam_ape_rot_unlim, sig_sc3_cam, 'unbounded', save, save_path_sc3_cam)
# APE_sigma_norm('rotation', sc3_cam_ape_rot_lim, sig_sc3_cam, 'bounded', save, save_path_sc3_cam)

# nbr_iterations_sigma(sc3_cam_iter, sig_sc3_cam, save, save_path_sc3_cam)


## Scenario 3 bis: Influence of sig on Gaussian noise (camera position + points) => PO for first 30 keyframes
sig_sc3bis = np.linspace(0, 0.15, 11)
sc3bis_ape_trans_unlim, sc3bis_ape_trans_lim = [], []
sc3bis_ape_rot_unlim, sc3bis_ape_rot_lim = [], []
sc3bis_iter = []
for seq in sequences:
    sc3bis_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3bis/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc3bis_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3bis/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc3bis_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3bis/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc3bis_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3bis/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    # sc3bis_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 3bis/{}/nbr_iterations_{}.npy'.format(seq, seq)))

save = 0
save_path_sc3bis = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 3bis/'


# APE_sigma('translation', sc3bis_ape_trans_unlim, sig_sc3bis, 'unbounded', save, save_path_sc3bis)
# APE_sigma('translation', sc3bis_ape_trans_lim, sig_sc3bis, 'bounded', save, save_path_sc3bis)

# APE_sigma('rotation', sc3bis_ape_rot_unlim, sig_sc3bis, 'unbounded', save, save_path_sc3bis)
# APE_sigma('rotation', sc3bis_ape_rot_lim, sig_sc3bis, 'bounded', save, save_path_sc3bis)

# APE_sigma_norm('translation', sc3bis_ape_trans_unlim, sig_sc3bis, 'unbounded', save, save_path_sc3bis)
# APE_sigma_norm('translation', sc3bis_ape_trans_lim, sig_sc3bis, 'bounded', save, save_path_sc3bis)

# APE_sigma_norm('rotation', sc3bis_ape_rot_unlim, sig_sc3bis, 'unbounded', save, save_path_sc3bis)
# APE_sigma_norm('rotation', sc3bis_ape_rot_lim, sig_sc3bis, 'bounded', save, save_path_sc3bis)

# nbr_iterations_sigma(sc3bis_iter, sig_sc3bis, save, save_path_sc3bis)

# Compare result of sc2 to sc3
save = 0
save_path_sc2sc3 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Comparison/Sc2_Sc3/'

# APE_sigma('translation', sc2_ape_trans_unlim, sig_sc2, 'unbounded, Scenario 2', save, save_path_sc2sc3, lim = [-0.01, 0.16, 0, 2.5])
# APE_sigma('translation', sc2_ape_trans_lim, sig_sc2, 'bounded, Scenario 2', save, save_path_sc2sc3, lim = [-0.01, 0.16, 0, 2.5])

# APE_sigma('rotation', sc2_ape_rot_unlim, sig_sc2, 'unbounded, Scenario 2', save, save_path_sc2sc3, lim = [-0.05, 0.16, 0, 25])
# APE_sigma('rotation', sc2_ape_rot_lim, sig_sc2, 'bounded, Scenario 2', save, save_path_sc2sc3, lim = [-0.05, 0.16, 0, 16.5])



# APE_sigma('translation', sc3_ape_trans_unlim, sig_sc3, 'unbounded, Scenario 3', save, save_path_sc2sc3, lim = [-0.01, 0.16, 0, 2.5])
# APE_sigma('translation', sc3_ape_trans_lim, sig_sc3, 'bounded, Scenario 3', save, save_path_sc2sc3, lim = [-0.01, 0.16, 0, 2.5])

# APE_sigma('rotation', sc3_ape_rot_unlim, sig_sc3, 'unbounded, Scenario 3', save, save_path_sc2sc3, lim = [-0.05, 0.16, 0, 25])
# APE_sigma('rotation', sc3_ape_rot_lim, sig_sc3, 'bounded, Scenario 3', save, save_path_sc2sc3, lim = [-0.05, 0.16, 0, 16.5])

# nbr_iterations_sigma(sc2_iter, sig_sc2, save, save_path_sc2sc3, '_scenario_2', ': Scenario 2', lim = [-0.05, 0.16, 0, 240])
# nbr_iterations_sigma(sc3_iter, sig_sc3, save, save_path_sc2sc3, '_scenario_3', ': Scenario 3', lim = [-0.05, 0.16, 0, 240])





## Scenario 4: Influence of mu on Random Walk noise (camera position only)
mu_sc4 = np.linspace(0, 0.0025, 11)
sc4_ape_trans_unlim, sc4_ape_trans_lim = [], []
sc4_ape_rot_unlim, sc4_ape_rot_lim = [], []
sc4_iter, sc4_timing = [], []

for seq in sequences:
    sc4_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc4_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc4_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc4_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc4_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 4/{}/timings_{}.npy'.format(seq, seq)))


save = 0
save_path_sc4 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 4/'

# APE_mu('translation',sc4_ape_trans_unlim, mu_sc4, 'unbounded', save, save_path_sc4)
# APE_mu('translation', sc4_ape_trans_lim, mu_sc4, 'bounded', save, save_path_sc4)

# APE_mu('rotation', sc4_ape_rot_unlim, mu_sc4, 'unbounded', save, save_path_sc4)
# APE_mu('rotation', sc4_ape_rot_lim, mu_sc4, 'bounded', save, save_path_sc4)

# APE_mu_norm('translation', sc4_ape_trans_unlim, mu_sc4, 'unbounded', save, save_path_sc4)
# APE_mu_norm('translation', sc4_ape_trans_lim, mu_sc4, 'bounded', save, save_path_sc4)

# APE_mu_norm('rotation', sc4_ape_rot_unlim, mu_sc4, 'unbounded', save, save_path_sc4)
# APE_mu_norm('rotation', sc4_ape_rot_lim, mu_sc4, 'bounded', save, save_path_sc4)


# nbr_iterations_mu(sc4_iter, mu_sc4, save, save_path_sc4)
# extract_time(sc4_timing, mu_sc4)

## Scenario 5: Influence of sigma on Random Walk noise applied to points
sig_sc5 = np.linspace(0, 0.15, 11)
sc5_ape_trans_unlim, sc5_ape_trans_lim = [], []
sc5_ape_rot_unlim, sc5_ape_rot_lim = [], []
sc5_iter, sc5_timing = [], []

for seq in sequences:
    sc5_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc5_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc5_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc5_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc5_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 5/{}/timings_{}.npy'.format(seq, seq)))


save = 0
save_path_sc5 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 5/'

# APE_sigma('translation', sc5_ape_trans_unlim, sig_sc5, 'unbounded', save, save_path_sc5)
# APE_sigma('translation', sc5_ape_trans_lim, sig_sc5, 'bounded', save, save_path_sc5)

# APE_sigma('rotation', sc5_ape_rot_unlim, sig_sc5, 'unbounded', save, save_path_sc5)
# APE_sigma('rotation', sc5_ape_rot_lim, sig_sc5, 'bounded', save, save_path_sc5)

# APE_sigma_norm('translation', sc5_ape_trans_unlim, sig_sc5, 'unbounded', save, save_path_sc5)
# APE_sigma_norm('translation', sc5_ape_trans_lim, sig_sc5, 'bounded', save, save_path_sc5)

# APE_sigma_norm('rotation', sc5_ape_rot_unlim, sig_sc5, 'unbounded', save, save_path_sc5)
# APE_sigma_norm('rotation', sc5_ape_rot_lim, sig_sc5, 'bounded', save, save_path_sc5)

# nbr_iterations_sigma(sc5_iter, sig_sc5, save, save_path_sc5)
# extract_time(sc5_timing, sig_sc5)


## Scenario 6: Influence of sigma on Random Walk noise applied to points
sig_sc6 = np.linspace(0, 0.15, 11)
sc6_ape_trans_unlim, sc6_ape_trans_lim = [], []
sc6_ape_rot_unlim, sc6_ape_rot_lim = [], []
sc6_iter, sc6_timing = [], []

# sequences = ['MH01', 'MH02', 'MH03']
for seq in sequences:
    sc6_ape_trans_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_unlim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_trans_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_lim_tran_{}.npy'.format(seq, seq)))
    sc6_ape_rot_unlim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_unlim_rot_{}.npy'.format(seq, seq)))
    sc6_ape_rot_lim.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/APE_lim_rot_{}.npy'.format(seq, seq)))
    sc6_iter.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/nbr_iterations_{}.npy'.format(seq, seq)))
    sc6_timing.append(np.load('/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/timings_{}.npy'.format(seq, seq)))

save = 0
save_path_sc6 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Scenario 6/'

# APE_sigma('translation', sc6_ape_trans_unlim, sig_sc6, 'unbounded', save, save_path_sc6)
# APE_sigma('translation', sc6_ape_trans_lim, sig_sc6, 'bounded', save, save_path_sc6)

# APE_sigma('rotation', sc6_ape_rot_unlim, sig_sc6, 'unbounded', save, save_path_sc6)
# APE_sigma('rotation', sc6_ape_rot_lim, sig_sc6, 'bounded', save, save_path_sc6)

# APE_sigma_norm('translation', sc6_ape_trans_unlim, sig_sc6, 'unbounded', save, save_path_sc6)
# APE_sigma_norm('translation', sc6_ape_trans_lim, sig_sc6, 'bounded', save, save_path_sc6)

# APE_sigma_norm('rotation', sc6_ape_rot_unlim, sig_sc6, 'unbounded', save, save_path_sc6)
# APE_sigma_norm('rotation', sc6_ape_rot_lim, sig_sc6, 'bounded', save, save_path_sc6)

# nbr_iterations_sigma(sc6_iter, sig_sc6, save, save_path_sc6)
extract_time(sc6_timing, sig_sc6)

# Compare result of sc5 to sc6
save = 0
save_path_sc5sc6 = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Sensitivity/SABA/Comparison/Sc5_Sc6/'


# APE_sigma('translation', sc5_ape_trans_unlim, sig_sc5, 'unbounded, Scenario 5', save, save_path_sc5sc6, lim = [-0.01, 0.16, 0, 2.5])
# APE_sigma('translation', sc5_ape_trans_lim, sig_sc5, 'bounded, Scenario 5', save, save_path_sc5sc6, lim = [-0.01, 0.16, 0, 2.5])

# APE_sigma('rotation', sc5_ape_rot_unlim, sig_sc5, 'unbounded, Scenario 5', save, save_path_sc5sc6, lim = [-0.05, 0.16, 0, 16])
# APE_sigma('rotation', sc5_ape_rot_lim, sig_sc5, 'bounded, Scenario 5', save, save_path_sc5sc6, lim = [-0.05, 0.16, 0, 16.5])


# APE_sigma('translation', sc6_ape_trans_unlim, sig_sc6, 'unbounded, Scenario 6', save, save_path_sc5sc6, lim = [-0.01, 0.16, 0, 2.5])
# APE_sigma('translation', sc6_ape_trans_lim, sig_sc6, 'bounded, Scenario 6', save, save_path_sc5sc6, lim = [-0.01, 0.16, 0, 2.5])

# APE_sigma('rotation', sc6_ape_rot_unlim, sig_sc6, 'unbounded, Scenario 6', save, save_path_sc5sc6, lim = [-0.05, 0.16, 0, 16])
# APE_sigma('rotation', sc6_ape_rot_lim, sig_sc6, 'bounded, Scenario 6', save, save_path_sc5sc6, lim = [-0.05, 0.16, 0, 16.5])

# nbr_iterations_sigma(sc5_iter, sig_sc5, save, save_path_sc5sc6, '_scenario_5', ': Scenario 5', lim = [-0.05, 0.16, 0, 250])
# nbr_iterations_sigma(sc6_iter, sig_sc6, save, save_path_sc5sc6, '_scenario_6', ': Scenario 6', lim = [-0.05, 0.16, 0, 250])






