import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import Shannon_entropy

def display_evaluation(method: str, alpha: float,
                       num_obs: int, N: int, N_envs: int,  
                       DICT: dict, LOADED: bool,
                       num_demo_types: int=4, num_types: int=4, n_buttons: int=20,
                       save: bool=True, saving_name: str=None) -> None:
    demo_colors = ['m', 'red', 'orange', 'pink']

    if LOADED:
        alpha = str(alpha)

    fig0 = plt.figure(figsize=(15,5))
    # Mean over the type of learner of the total reward on trajectory of size 20  (after seen the demo chosen by the teacher)
    all_evals = np.array([DICT[method][alpha][str(type)]['rewards'] if LOADED else DICT[method][alpha][type]['rewards'] for type in range(num_types)]).mean(axis=0)
    mean = np.mean(all_evals, axis=0)
    std = np.std(all_evals)

    plt.plot(mean, label=f'{method}', color='saddlebrown')
    plt.ylim(0, 21)
    plt.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std/np.sqrt(N * N_envs), alpha=0.2, color='saddlebrown')
    plt.plot(np.arange(num_obs), [20] * num_obs, c='k', label="Max", ls='--')
    
    # Baseline MAP --> Uniform
    if method == 'MAP':
        baseline = 'Uniform'
        if baseline in DICT.keys():
            all_evals = np.array([DICT[baseline][alpha][str(type)]['rewards'] if LOADED else DICT[baseline][alpha][type]['rewards'] for type in range(num_types)]).mean(axis=0)
            mean = np.mean(all_evals, axis=0)
            std = np.std(all_evals)
            plt.plot(mean, label=f'Baseline ({baseline})', color='crimson', ls='--')
            plt.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std/np.sqrt(N * N_envs), alpha=0.2, color='crimson')

    plt.title(fr' Mean total reward over all the type of learner'  + f'\n $\mathbf{{{method}}}$ teacher, cost parameter alpha={alpha}')
    plt.xlabel('Size of the learner trajectory observed by the teacher')
    plt.ylabel('Learner reward')
    plt.legend()

    fig1 = plt.figure(figsize=(15,5))
    # Learner total reward on trajectory of size 20 (after seen the demo chosen by the teacher for each type of learner)
    fig1.add_subplot(1,2,1)
    for type in range(num_types):
        if LOADED:
            type = str(type)
        all_evals = np.array(DICT[method][alpha][type]['rewards'])
        mean = np.mean(all_evals, axis=0)
        std = np.std(all_evals)

        plt.plot(mean, label=f'type = {type}')
        plt.ylim(0, 21)
        plt.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std / np.sqrt(N * N_envs), alpha=0.2)
    plt.plot(np.arange(num_obs), [20] * num_obs, c='k', label="Max", ls='--')
    plt.title(fr'Learner total reward per type'  + f'\n $\mathbf{{{method}}}$ teacher, cost parameter alpha={alpha}' + f'\n model {saving_name}')

    plt.xlabel('Size of the learner trajectory observed by the teacher')
    plt.ylabel('Learner reward')
    plt.legend()

    # Teacher uncertainty
    if method in ['MAP', 'Bayesian']:
        fig1.add_subplot(1,2,2)
        for type in range(num_types):
            if LOADED:
                type = str(type)
            all_evals = np.array(DICT[method][alpha][type]['uncertainty'])
            mean = np.mean(all_evals, axis=0)
            std = np.std(all_evals)

            plt.plot(mean, label=f'type = {type}')
            plt.ylim(- 0.5 , Shannon_entropy(np.ones(num_types) / num_types) + 0.5)
            plt.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std/np.sqrt(N * N_envs), alpha=0.2)
        plt.plot(np.arange(num_obs), [0] * num_obs, c='k', label="Min", ls='--')
        plt.title(fr'Teacher uncertainty per type' + f'\n $\mathbf{{{method}}}$ teacher, cost parameter alpha={alpha}')

        plt.xlabel('Size of the learner trajectory observed by the teacher')
        plt.ylabel('Teacher uncertainty (Shannon entropy)')
        plt.legend()

    # Repartition btw type of demonstrations shown
    fig2, axes = plt.subplots(1, 4, figsize=(25,5))
    for type in range(num_types):
        ax = axes[type]
        if LOADED:
            type = str(type)
        for demo_type in range(num_demo_types):
            demo_rep = np.array(DICT[method][alpha][type]['demo'])
            if demo_type == 0:
                prop_demo = np.array(demo_rep == n_buttons)
            else:
                prop_demo = np.array(demo_rep == demo_type)
            mean = np.mean(prop_demo, axis=0)
            std = np.std(prop_demo)

            ax.plot(mean, label=f'demo type = {demo_type}', color=demo_colors[demo_type])
            ax.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N + N_envs), mean - 1.96 * std/np.sqrt(N + N_envs), alpha=0.2, color=demo_colors[demo_type])
        ax.plot(np.arange(num_obs), [1] * num_obs, c='k', label="Max", ls='--')

        ax.set_title(f'% of each demo type for learner of $\mathbf{{type}}$ ' + fr'$\mathbf{{{type}}}$' + f'\n {method} teacher, cost parameter alpha={alpha}')
        ax.set_xlabel('Size of the learner trajectory observed by the teacher')
        ax.set_ylabel('Proportion')
        ax.legend()

    # Teacher regret on the cost of the demonstration he showed
    fig3, axes = plt.subplots(1, 4, figsize=(25,5))
    for type in range(num_types):
        ax = axes[type]
        if LOADED:
            type = str(type)
        all_regrets = np.array(DICT[method][alpha][type]['teacher_regret'])
        mean = np.mean(all_regrets, axis=0)
        std = np.std(all_regrets)

        ax.plot(mean, label=f'type = {type}', color='c')
        ax.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std/np.sqrt(N * N_envs), alpha=0.2, color='c')
        ax.set_ylim( - (float(alpha) * (n_buttons - 1) + 0.02), float(alpha) * (n_buttons - 1) + 0.02)
        ax.plot(np.arange(num_obs), [0] * num_obs, c='k', ls='--')

        # Baseline Bayesian --> Opt_non_adaptive
        if method == 'Bayesian':
            baseline = 'Opt_non_adaptive'
            all_regrets = np.array(DICT[baseline][alpha][type]['teacher_regret'])
            mean = np.mean(all_regrets, axis=0)
            std = np.std(all_regrets)

            ax.plot(mean, label=f'Baseline ({baseline})', color='crimson', ls='--')
            ax.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std/np.sqrt(N * N_envs), alpha=0.2, color='crimson')

        ax.set_title(f'Teacher cost regret for learner of $\mathbf{{type}}$ ' + fr'$\mathbf{{{type}}}$' +  f'\n {method} teacher, cost parameter alpha={alpha}')
        ax.set_xlabel('Size of the learner trajectory observed by the teacher')
        ax.set_ylabel('Cost regret')
        ax.legend()

    # Super figure
    figs = [fig0, fig1, fig2, fig3]
    saving_names = ['mean', 'per_type', 'chosen_demo', 'teacher_regret']
    if save:
        if method in ['MAP', 'Bayesian', 'Uniform', 'Opt_non_adaptive']:
            for ii, fig in enumerate(figs):
                fig.savefig(f'./bayesian_ToM/figures/{method}/eval_{saving_names[ii]}_{method}_{alpha}.png')
        elif method in ['ToMNet']:
            for ii, fig in enumerate(figs):
                fig.savefig(f'./neural_network_ToM/figures/{method}/eval_{saving_names[ii]}_{method}_{alpha}_{saving_name}.png')


def display_utility(alpha: float,
                    num_obs: int, N: int, N_envs: int,
                    DICT: dict, LOADED: bool,
                    num_types: int=4, n_buttons: int=20,
                    saving_name: str=None) -> None:
    fig = plt.figure(figsize=(15,5))
    method_values = ['MAP', 'Bayesian', 'Uniform', 'Opt_non_adaptive', 'Oracle', 'ToMNet']
    colors = ['orangered', 'mediumvioletred', 'darkturquoise', 'royalblue', 'darkgreen', 'pink']
    for ii,method in enumerate(method_values):
        util = []
        for type in range(num_types):
            best_cost = alpha * n_buttons if type == 0 else alpha * type
            all_rewards = np.array([DICT[method][str(alpha)][str(type)]['rewards'] if LOADED else DICT[method][alpha][type]['rewards'] for type in range(num_types)]).mean(axis=0)
            all_cost = np.array([DICT[method][str(alpha)][str(type)]['teacher_regret'] + best_cost if LOADED else DICT[method][alpha][type]['rewards'] + best_cost for type in range(num_types)]).mean(axis=0)

            all_util = all_rewards / n_buttons - all_cost
            util.append(all_util)
            
        all = np.mean(util, axis=0)
        mean = np.mean(all, axis=0)
        std = np.std(all)
        plt.plot(mean, label=f'{method}', color=colors[ii])
        plt.fill_between(np.arange(num_obs), mean + 1.96 * std / np.sqrt(N * N_envs), mean - 1.96 * std/np.sqrt(N * N_envs), alpha=0.2, color=colors[ii])
    
    if alpha in [0.01, 0.02]:
        plt.ylim(0.5, 1)
    
    plt.xlabel('Size of the learner trajectory observed by the teacher')
    plt.ylabel('Utility')
    plt.title(f'Mean utility over all the type of learner (95% c.i) \n cost parameter alpha={alpha} \n ToMNet model {saving_name}')
    plt.legend()
    
    fig.savefig(f'./neural_network_ToM/figures/all_utilities_{alpha}_{saving_name}.png');

def display_utility_hist(alpha: float,
                    num_obs: int, N: int, N_envs: int,
                    DICT: dict, LOADED: bool,
                    num_types: int=4, n_buttons: int=20,
                    saving_name: str=None) -> None:
    fig = plt.figure(figsize=(20, 6))
    method_values = ['ToMNet', 'Bayesian', 'Uniform', 'Opt_non_adaptive', 'Oracle']
    colors = ['pink', 'mediumvioletred', 'darkturquoise', 'royalblue', 'darkgreen']
    
    points = [0, 3, 5, 10, 20, 49]

    handles = []  # List to store legend handles
    for ii,method in enumerate(method_values):
        handle = mpatches.Patch(color=colors[ii], label=method)
        handles.append(handle)
        
    for ii, n in enumerate(points):
        ax = fig.add_subplot(1, len(points), ii + 1)
        utilities = []
        errors = []
        for method in method_values:
            util = []
            for type in range(num_types):
                best_cost = alpha * n_buttons if type == 0 else alpha * type
                all_rewards = np.array([DICT[method][str(alpha)][str(type)]['rewards'] if LOADED else DICT[method][alpha][type]['rewards'] for type in range(num_types)]).mean(axis=0)
                all_cost = np.array([DICT[method][str(alpha)][str(type)]['teacher_regret'] + best_cost if LOADED else DICT[method][alpha][type]['rewards'] + best_cost for type in range(num_types)]).mean(axis=0)
                all_util = all_rewards / n_buttons - all_cost
                util.append(all_util)
            
            all = np.mean(util, axis=0)
            mean = np.mean(all, axis=0)
            std_utils = 1.96 * np.std(all, axis=0) / np.sqrt(N * N_envs)
            utilities.append(mean[n])
            errors.append(std_utils[n])

        ax.bar(range(len(method_values)), utilities, width=1., yerr=errors, color=colors)
        ax.set_ylim(0.60, 0.95)
        ax.set_title(f'n = {n}')
        ax.set_xticks([])
    
    fig.suptitle(f'Mean utility for different sizes $n$ of the learner trajectory observed by the teacher \n cost parameter alpha={alpha}', fontsize='x-large')  # Set main title
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, - 0.1), ncol=len(method_values), fontsize='x-large')  # Add shared legend
    plt.tight_layout()  # Adjust the layout of subplots
    plt.show()

    fig.savefig(f'./neural_network_ToM/figures/all_utilities_hist_{alpha}_{saving_name}.png');


def display_utility_errorbar(alpha: float,
                    num_obs: int, N: int, N_envs: int,
                    DICT: dict, LOADED: bool,
                    num_types: int=4, n_buttons: int=20,
                    saving_name: str=None) -> None:
    fig = plt.figure(figsize=(20, 6))
    method_values = ['ToMNet', 'Bayesian', 'Uniform', 'Opt_non_adaptive', 'Oracle']
    colors = ['pink', 'mediumvioletred', 'darkturquoise', 'royalblue', 'darkgreen']
    
    points = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 49]
    
    for ii, n in enumerate(points):
        utilities = []
        errors = []
        for jj,method in enumerate(method_values):
            util = []
            for type in range(num_types):
                best_cost = alpha * n_buttons if type == 0 else alpha * type
                all_rewards = np.array([DICT[method][str(alpha)][str(type)]['rewards'] if LOADED else DICT[method][alpha][type]['rewards'] for type in range(num_types)]).mean(axis=0)
                all_cost = np.array([DICT[method][str(alpha)][str(type)]['teacher_regret'] + best_cost if LOADED else DICT[method][alpha][type]['rewards'] + best_cost for type in range(num_types)]).mean(axis=0)
                all_util = all_rewards / n_buttons - all_cost
                util.append(all_util)
            
            all = np.mean(util, axis=0)
            mean = np.mean(all, axis=0)
            std_utils = 1.96 * np.std(all, axis=0) / np.sqrt(N * N_envs)
            utilities.append(mean[n])
            errors.append(std_utils[n])

            if method in ['ToMNet', 'Bayesian', 'Uniform']:
                plt.errorbar([n], [mean[n]], yerr=std_utils[n], color=colors[jj], fmt="o")
                if ii == len(points) - 1:
                    plt.errorbar([n], [mean[n]], yerr=std_utils[n], color=colors[jj], fmt="o", label=method)
            else:
                if ii == len(points) - 1:
                    plt.plot(mean, label=f'{method}', color=colors[jj])
                else:
                    plt.plot(mean, color=colors[jj])
    
    fig.suptitle(f'Mean utility for different sizes $n$ of the learner trajectory observed by the teacher \n cost parameter alpha={alpha}', fontsize='x-large')  # Set main title
    plt.legend()
    plt.xlabel('Size $n$ of the learner trajectory observed by the teacher \n i.e. amount of information the teacher has about the learner')
    plt.ylabel('Utility')
    plt.ylim(0.5)
    plt.xticks(points)
    plt.show()

    fig.savefig(f'./neural_network_ToM/figures/all_utilities_errorbar_{alpha}_{saving_name}.png');


def display_utility_errorbar_split(alpha: float,
                    num_obs: int, N: int, N_envs: int,
                    DICT: dict, LOADED: bool,
                    num_types: int=4, n_buttons: int=20,
                    saving_name: str=None) -> None:
    fig = plt.figure(figsize=(20, 6))
    method_values = ['ToMNet', 'Bayesian', 'Uniform', 'Opt_non_adaptive', 'Oracle']
    colors = ['pink', 'mediumvioletred', 'darkturquoise', 'royalblue', 'darkgreen']
    
    points = [0, 3, 5, 10, 20, 49]

    handles = []  # List to store legend handles
    for ii,method in enumerate(method_values):
        handle = mpatches.Patch(color=colors[ii], label=method)
        handles.append(handle)
        
    for ii, n in enumerate(points):
        ax = fig.add_subplot(1, len(points), ii + 1)
        utilities = []
        errors = []
        for jj,method in enumerate(method_values):
            util = []
            for type in range(num_types):
                best_cost = alpha * n_buttons if type == 0 else alpha * type
                all_rewards = np.array([DICT[method][str(alpha)][str(type)]['rewards'] if LOADED else DICT[method][alpha][type]['rewards'] for type in range(num_types)]).mean(axis=0)
                all_cost = np.array([DICT[method][str(alpha)][str(type)]['teacher_regret'] + best_cost if LOADED else DICT[method][alpha][type]['rewards'] + best_cost for type in range(num_types)]).mean(axis=0)
                all_util = all_rewards / n_buttons - all_cost
                util.append(all_util)
            
            all = np.mean(util, axis=0)
            mean = np.mean(all, axis=0)
            std_utils = 1.96 * np.std(all, axis=0) / np.sqrt(N * N_envs)
            utilities.append(mean[n])
            errors.append(std_utils[n])

            if method in ['ToMNet', 'Bayesian', 'Uniform']:
                ax.errorbar([n], [mean[n]], yerr=std_utils[n], color=colors[jj], fmt="o", markersize=10)
            else:
                ax.plot([n-0.01,n+0.01], [mean[n], mean[n]], color=colors[jj])
        ax.set_ylim(0.65, 0.95)
        ax.set_title(f'n = {n}')
        ax.set_xticks([])
    
    fig.suptitle(f'Mean utility for different sizes $n$ of the learner trajectory observed by the teacher \n cost parameter alpha={alpha}', fontsize='x-large')  # Set main title
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, - 0.1), ncol=len(method_values), fontsize='x-large')  # Add shared legend
    plt.tight_layout()  # Adjust the layout of subplots
    plt.show()

    fig.savefig(f'./neural_network_ToM/figures/all_utilities_errorbar_split_{alpha}_{saving_name}.png');