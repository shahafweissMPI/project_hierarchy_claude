import helperFunctions as hf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

videopath=r"\\gpfs.corp.brain.mpg.de\stem\data\Data_raw\enrico_data\behavioral record\record\escape_single_housed_females"
datapath=r"C:\Users\kernt\Documents\awesome_files\boris_output"

plt.close('all')

run_checks=True
loomtimes=[0.  , 1.13, 2.26, 3.39, 4.52] #gives loomstarts in frames
max_delay=5*50

def check_start_stop(start_stop_vector):
    "checks whether starts and stop are alternating"
    boolean_vector=np.zeros(start_stop_vector.shape)
    boolean_vector[start_stop_vector=='START']=0
    boolean_vector[start_stop_vector=='STOP']=1
    boolean_vector=boolean_vector.astype(int)
    
    adjacent_different=np.all(np.logical_xor(boolean_vector[:-1], boolean_vector[1:]))
    even_length=len(boolean_vector) % 2 == 0
    alternating=adjacent_different and even_length
    
    return alternating

def get_trialstarts(data):
    data_s=[entry/60 for entry in data]
    trialstarts=[len(entries) for entries in data_s]    
    trialstarts=np.cumsum(trialstarts)    
    trialstarts=np.insert(trialstarts,0,0)[:-1]
    
    y_trialstarts=[entry[0] for entry in data_s]
    conc=np.concatenate(data_s)
    return trialstarts, y_trialstarts, conc 

animal_colors=['indianred','goldenrod','olive']
dark_animal_colors=['maroon','darkgoldenrod','darkolivegreen']
animallist=[entry for entry in os.listdir(f'{videopath}') if '.' not in entry]
# animallist=['vsm16245_059']
for ia, animal in enumerate(animallist):
    sessionlist=np.sort(os.listdir(f'{videopath}\{animal}'))
    # sessionlist=['230920']

    #preallocate
    an_first_reaction=[]
    an_first_behaviour=[]
    an_escape_latency=[]
    an_loom_escape=[]
    an_loom_intensity=[]
    
    for session in sessionlist:

        #Load data
        loomframes_bonsai=np.genfromtxt(f'{videopath}\{animal}\{session}\loomframes.csv',delimiter=',')[:,0]
        A=animal[-1]
        S=session[-2:]
        try:
            data=pd.read_csv(f'{datapath}\A{A}S{S}.csv')
        except FileNotFoundError:
            print('no file, skipping')
            continue
        
        
        #Get relevant columns/ data
        duration_s=data['Media duration (s)'].iloc[0]
        duration_min=duration_s/60
        behaviours=data['Behavior'].to_numpy()
        frames=data['Image index'].to_numpy()
        start_stop=data['Behavior type'].to_numpy()
        modifier=data['Modifier #1'].to_numpy()
        os.chdir(f'{videopath}\{animal}\{session}')
        for file in glob.glob('*looms_timestamp*', recursive=True):
            loom_path = file
        csv_looms=pd.read_csv(loom_path, header=None)
        loom_intensity=csv_looms[0].str.extract(r'(\d+)').to_numpy() #low number = low contrast
        loom_intensity[loom_intensity=='2']='9'
        #Relabel freezing
        behaviours[behaviours=='freeze']='immobility'
        
        
        #Run checks
        loomframes_ind=np.where(behaviours=='loom')[0]
        if run_checks:
            #Check whether looms were annotated correctly            
            loomframes_boris=frames[loomframes_ind]
            if not np.array_equal(loomframes_boris,loomframes_bonsai):
                print(f'\nFucking hell! A{A}S{S} is causing problems')
                print('bonsai')
                print(loomframes_bonsai)
                print('boris')
                print(loomframes_boris)
                print()
            
            #cHECK STARTS AND STOPS
            for behav in ['startle','immobility','escape']:
                if not check_start_stop(start_stop[behaviours==behav]):
                    print('ALARM! there is something wrong with the start/stop annotation')
                    print(f'{animal,session}')
            
        
        #get loom behaviour
        num_looms=len(loomframes_ind)
        loom_escape=[] # 0=no escape 1=failed escape 2=successful escape
 
        
        #preallocate
        first_reaction=[]
        first_behaviour=[]
        escape_latency=[]
        
        for iloom,loomind in enumerate(loomframes_ind):
                        
            #Is there a reaction?
            escape_latency.append(np.nan)
            loom_escape.append(0)
            if (len(behaviours)!=loomind+1) and (behaviours[loomind+1] != 'loom'): 
                #the current entry is not the last entry AND the next entry is not a loom

                #Is the reaction before the loom?
                distance_before=np.abs(frames[loomind-1]-frames[loomind])
                distance_next=np.abs(frames[loomind+1]-frames[loomind])
                if distance_before<= 250:
                    first_reaction.append(frames[loomind-1]-frames[loomind])
                    first_behaviour.append(behaviours[loomind-1])
                elif distance_next<max_delay:
                  first_reaction.append(frames[loomind+1]-frames[loomind])
                  first_behaviour.append(behaviours[loomind+1])  

                
                
                #get all the behaviours between this and the next loom
                if iloom+1!=len(loomframes_ind):
                    next_loomind=loomframes_ind[iloom+1] #= inter loom interval
                else:
                    next_loomind=len(frames)-1
                
                #is there an escape?
                if 'escape' in behaviours[loomind:next_loomind]:

                    # get escape latency
                    if 'start turn' in behaviours[loomind:next_loomind]:
                        escape_signal='start turn'
                    elif 'escape_jumpstart' in behaviours[loomind:next_loomind]:
                        escape_signal='escape_jumpstart'
                    else:
                        print(f'REVISIT {animal,session}, add jumpstart')
                        print(frames[loomind])
                        print()
                        escape_signal='escape'
                    an_escape_signals=np.where(behaviours==escape_signal)[0]
                    escape_bool=(an_escape_signals>loomind) & (an_escape_signals<next_loomind)
                    escapesignal_ind=an_escape_signals[escape_bool][0]
                    loop_latency=frames[escapesignal_ind]-frames[loomind]
                                            
                    #Was the escape successful?
                    if ('success' in modifier[loomind:next_loomind]) and (loop_latency<max_delay):
                        escape_latency[-1]=loop_latency
                        loom_escape[-1]=2
                    elif ('failed' in modifier[loomind:next_loomind]) and (loop_latency<max_delay):
                        loom_escape[-1]=1
                       
                # # Events Plot
                # plt.figure()
                
                # #get Length vector
                # behaviour_starts


        
        an_first_reaction.append(np.array(first_reaction))
        an_first_behaviour.append(np.array(first_behaviour))
        an_escape_latency.append(np.array(escape_latency))
        an_loom_escape.append(np.array(loom_escape))
        an_loom_intensity.append(np.squeeze(loom_intensity))
    an_loom_intensity=np.hstack(an_loom_intensity).astype(int)
    an_loom_escape=np.hstack(an_loom_escape)
    an_escape_latency=np.hstack(an_escape_latency)/50

    
    # calculate by contrast    
    intensities=np.unique(an_loom_intensity)
    escape_prob_intense=[]
    escape_latency_intense=[]
    trialnum=[]
    for intensity in intensities:
        index=np.where(an_loom_intensity==intensity)[0]
        trialnum.append(len(index))
        #Escape probability
        num_looms=len(index)
        num_escapes=np.sum(an_loom_escape[index]==2 )
        escape_probability=num_escapes/num_looms
        escape_prob_intense.append(escape_probability)
        
        #Escape latency
        escape_latency_intense.append(np.nanmean(an_escape_latency[index]))
        
        
    escape_prob_intense=np.array(escape_prob_intense)
    
    #%% PLOT the results
    
    #Escape probability
    plt.figure('escape_probability')
    plt.plot(intensities*10,escape_prob_intense, label=animal, color=animal_colors[ia])
    
    plt.xlabel('contrast')
    plt.ylabel('escape probability')
    plt.title('escape probability by contrast (only successful)')
    plt.legend()
    
    #trials per contrast
    plt.figure('trials per contrast')
    plt.xlabel('contrast')
    plt.ylabel('Number of trials')
    plt.plot(intensities*10,trialnum, label=animal, color=animal_colors[ia])
    plt.legend()
    
    
    # plt.figure('escape_probability_success')
    # plt.plot(an_succ_escape_probability, label=animal, color=animal_colors[ia])
    # plt.xlabel('sessions')
    # plt.ylabel('escape probability')
    # plt.title('escape probability, only successful escapes')
    # plt.legend()
    
    #Latencies plot
    plt.figure('first reaction')
    
    trialstarts, y_trialstarts, conc_first_reaction =get_trialstarts(an_first_reaction)
    
    plt.plot(conc_first_reaction, label=animal, color=animal_colors[ia])
    plt.plot(trialstarts,y_trialstarts,'o', color=dark_animal_colors[ia])

    plt.xlabel('looms (new sessions marked by dots)')
    plt.ylabel('latency [s]')
    plt.title('latency of first reaction after loom')
    plt.legend()
    

    plt.figure('escape latency')
    
    # trialstarts, y_trialstarts, conc_escape_latency =get_trialstarts(an_escape_latency)
    
    # plt.plot(conc_escape_latency, label=animal, color=animal_colors[ia])
    # plt.plot(trialstarts,y_trialstarts,'o', color=dark_animal_colors[ia])
    
    plt.plot(intensities*10,escape_latency_intense, label=animal, color=animal_colors[ia])
    plt.xlabel('contrast')
    plt.ylabel('latency [s]')
    plt.title('latency of escape by contrast')
    plt.legend()          
              
    # Latencies Histogram      
    plt.figure()
    plt.title(f'{animal}')
    plt.hist(np.hstack(an_escape_latency),bins=50)
    plt.xlabel('escape latency [s]')
    for loom in loomtimes:
        plt.axvline(loom, color='r')
        

        

        
                    
                    
                   
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
  
        
