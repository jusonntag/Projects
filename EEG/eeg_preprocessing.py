import numpy as np
import pandas as pd
import mne
import os

class EEGNet_preprocessing:
    def __init__(self, high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample):
        '''
        high_pass = float // False
        low_pass = float // False
        window_begin = float
        window_end = float
        filterbank = linked list[[1,4,8,12, ...],[4,8,12, ...]] // False
        raw_dict = string with a '/' at the end
        '''
       
        if high_pass is False:
            self.high_pass = False
            self.low_pass = False
        elif type(high_pass) == 'float' or 'int':
            self.high_pass = high_pass
            self.low_pass = low_pass

        self.window_begin = window_begin
        self.window_end = window_end
        if filterbank is False:
            self.filterbank = filterbank
        elif type(filterbank) == 'list':
            self.low_pass = filterbank[1]
            self.high_pass = filterbank[0]
        if type(resample) == 'float' or 'int':
            resample_ratio = 512/resample
            self.resample = resample_ratio
        

        self.raw_dict = raw_dict
        self.save_dict = '/'.join(self.raw_dict.split('/')[:-2]) + '/preprocessed/'
        self.save_resting = '/'.join(self.raw_dict.split('/')[:-2])


    def get_datafiles(self):
        '''
        looks for all .set files and returns a dataframe with all paths (where raw file is loaded and preprocessed file will be saved) of it
        '''
        files = []
        print(self.raw_dict)
        for vp in os.listdir(self.raw_dict):
            for file in os.listdir(self.raw_dict + vp):
                if file.endswith('.set'):
                    files.append([self.raw_dict + vp + '/' + file, self.save_dict + vp + '/'])
        return pd.DataFrame(files, columns=['file_path', 'save_path'])


    def epoch2data(self, epoch_class, e):
        ''''
        turns mne epoch into numpy array
        '''
        data_epochs = epoch_class[e].get_data()
        label = [e for i in data_epochs]
        return data_epochs, label
    

    def load_data(self, file_path):
        ''''
        loads data depending on the file format
        '''
        if file_path.endswith('.set'):
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        elif file_path.endswith('.fif'):
            raw = mne.oi.read_raw_fif(file_path, preload = True)
        else:
            print('Error while trying to load file. Wrong Format?')
        return raw


    def get_resting(self, raw, events_from_annot, event_dict):
        ''''
        epochs data for eye resting state
        '''
        rest_dict = {}
        for key, value in event_dict.items():
            if key == '98' or key == '99':  # Check if the key is '98' or '99'
                rest_dict[key] = value 
        epochs = mne.Epochs(raw, events_from_annot, tmin = 2.5, tmax = 182.5, event_id = rest_dict, event_repeated = 'drop', preload = True, baseline = None)
        return epochs
    

    def apply_ica(self, raw):
        '''
        applys ICA to deicard artifacs
        then applys baselinecorretion from -0.2 to 0.0
        set M1 and M2 as reference 
        drops EOG channels
        '''
        eog_evoked = mne.preprocessing.create_eog_epochs(raw)
        eog_evoked.apply_baseline(baseline = (None, -0.2))
        ica = mne.preprocessing.ICA(n_components = 15, max_iter = 'auto', random_state = 42)
        ica.fit(raw)
        ica.exclude = []
        eog_incides, eog_scores = ica.find_bads_eog(raw)
        ica.exclude = eog_incides
        ica.apply(raw)
        raw.set_eeg_reference(ref_channels =  ['M1', 'M2'])
        raw.drop_channels(ch_names = ['VEOG', 'HEOG'])
        return raw


    def preprocess(self):
        '''
        function for precprcessing EEG data
        pereprocesses all data in of raw_dict with .set or .fif format
        applies filters as indicated
        '''
        save_list = []
        files_for_preprocessing = self.get_datafiles()
        for index, row in files_for_preprocessing.iterrows():
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            raw = self.load_data(row['file_path'])
            vp = row['file_path'].split('/')[-2]
            # apply notch filter to eleiminate flickering of light
            if self.low_pass > 49:
                raw = raw.notch_filter(50)
            raw.filter(self.high_pass, self.low_pass)
            # turn eog channels to eog in mne.info
            raw.set_channel_types({'VEOG': 'eog','HEOG': 'eog'})
            # apply monatge
            # montage = mne.channels.make_standard_montage('standard_1020')
            # raw.set_montage(montage)
            raw = raw.resample(sfreq=raw.info['sfreq']/self.resample)
            raw = self.apply_ica(raw)
            events_from_annot, event_dict = mne.events_from_annotations(raw)

            epochs = mne.Epochs(raw, events_from_annot, tmin = self.window_begin, tmax = self.window_end, event_id = event_dict, event_repeated = 'drop', preload = True, baseline = None)
            if '11' in event_dict:
                X_1, label_1 = self.epoch2data(epochs, '11')
                X_2, label_2 = self.epoch2data(epochs, '12')
                X_3, label_3 = self.epoch2data(epochs, '21')
                X_4, label_4 = self.epoch2data(epochs, '22')
                X_5, label_5 = self.epoch2data(epochs, '13')
                X_6, label_6 = self.epoch2data(epochs, '23')
            elif ' 11' in event_dict:
                X_1, label_1 = self.epoch2data(epochs, ' 11')
                X_2, label_2 = self.epoch2data(epochs, ' 12')
                X_3, label_3 = self.epoch2data(epochs, ' 21')
                X_4, label_4 = self.epoch2data(epochs, ' 22')
                X_5, label_5 = self.epoch2data(epochs, ' 13')
                X_6, label_6 = self.epoch2data(epochs, ' 23')
            else:
                print('erorr in:',event_dict)

            epochs = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis = 0 )
            label = np.concatenate([label_1, label_2, label_3, label_4, label_5, label_6], axis = 0)
            label = [item.strip() for item in label]

            # cheack and creates nes folder for preprocessed data of participant
            if not os.path.exists(row['save_path']):
                os.makedirs(row['save_path'])

            save_path_X = row['save_path'] + vp + '_X_'+ str(self.high_pass) + '-' + str(self.low_pass) + 'Hz_'+ str(abs(self.window_begin)+self.window_end) +'s_ica_128Hz_EEGNet.npy'
            save_path_Y = row['save_path'] + vp + '_Y_'+ str(self.high_pass) + '-' + str(self.low_pass) + 'Hz_'+ str(abs(self.window_begin)+self.window_end) +'s_ica_128Hz_EEGNet.npy'
            save_list.append([int(vp), save_path_X, save_path_Y])
            np.save(save_path_X, epochs)
            np.save(save_path_Y, label)
        save_list = pd.DataFrame(save_list, columns=['vp' ,'save_path_X_EEGNet', 'save_path_Y_EEGNet'])
        save_list = save_list.sort_values(by=['vp'])
        return save_list


class FBCNet_preprocessing(EEGNet_preprocessing):
    def __init__(self, high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample):
        super().__init__(high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample)
        self.low_pass = filterbank[1]
        self.high_pass = filterbank[0]


    def preprocess(self):
        '''
        function for precprcessing EEG data
        pereprocesses all data in of raw_dict with .set or .fif format
        applies filters as indicated
        '''        
        save_list = []
        files_for_preprocessing = self.get_datafiles()
        for index, row in files_for_preprocessing.iterrows():
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            raw = self.load_data(row['file_path'])
            vp = row['file_path'].split('/')[-2]

            # turn eog channels to eog in mne.info
            raw.set_channel_types({'VEOG': 'eog','HEOG': 'eog'})
            # apply monatge
            # montage = mne.channels.make_standard_montage('standard_1020')
            # raw.set_montage(montage)
            raw = raw.resample(sfreq=raw.info['sfreq']/self.resample)
            #raw = self.apply_ica(raw)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            epochs_fb = []
            for i,_ in enumerate(self.high_pass):
                raw.filter(self.high_pass[i], self.low_pass[i])

                epochs = mne.Epochs(raw, events_from_annot, tmin = self.window_begin, tmax = self.window_end, event_id = event_dict, event_repeated = 'drop', preload = True, baseline = None)
                
                if '11' in event_dict:
                    X_1, label_1 = self.epoch2data(epochs, '11')
                    X_2, label_2 = self.epoch2data(epochs, '12')
                    X_3, label_3 = self.epoch2data(epochs, '21')
                    X_4, label_4 = self.epoch2data(epochs, '22')
                    X_5, label_5 = self.epoch2data(epochs, '13')
                    X_6, label_6 = self.epoch2data(epochs, '23')
                elif ' 11' in event_dict:
                    X_1, label_1 = self.epoch2data(epochs, ' 11')
                    X_2, label_2 = self.epoch2data(epochs, ' 12')
                    X_3, label_3 = self.epoch2data(epochs, ' 21')
                    X_4, label_4 = self.epoch2data(epochs, ' 22')
                    X_5, label_5 = self.epoch2data(epochs, ' 13')
                    X_6, label_6 = self.epoch2data(epochs, ' 23')
                else:
                    print('erorr in:',event_dict)

                epochs = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis = 0 )
                epochs = np.expand_dims(epochs, axis = 3)
                label = np.concatenate([label_1, label_2, label_3, label_4, label_5, label_6], axis = 0)
                label = [item.strip() for item in label]
                #concat all filterbanks
                if len(epochs_fb) == 0:
                    epochs_fb = epochs
                else:
                    epochs_fb = np.concatenate((epochs_fb,epochs), axis = 3)

            epochs = epochs_fb
            
            # cheack and creates new folder for preprocessed data of participant
            if not os.path.exists(row['save_path']):
                os.makedirs(row['save_path'])

            save_path_X = row['save_path'] + vp + '_X_'+ str(self.high_pass[0]) + '-' + str(self.low_pass[-1]) + 'Hz-FB_' + str(abs(self.window_begin)+self.window_end) +'s_ica_128Hz_FBCNet.npy'
            save_path_Y = row['save_path'] + vp + '_Y_'+ str(self.high_pass[0]) + '-' + str(self.low_pass[-1]) + 'Hz-FB_' + str(abs(self.window_begin)+self.window_end) +'s_ica_128Hz_FBCNet.npy'
            save_list.append([int(vp), save_path_X, save_path_Y])
            np.save(save_path_X, epochs)
            np.save(save_path_Y, label)
        save_list = pd.DataFrame(save_list, columns=['vp' ,'save_path_X_FBCNet', 'save_path_Y_FBCNet'])
        save_list = save_list.sort_values(by=['vp'])
        return save_list
    

class NFEEG_preprocessing(EEGNet_preprocessing):
    def __init__(self, high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample):
        super().__init__(high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample)


    def apply_ica(self, raw):
        '''
        applys ICA to deicard artifacs
        then applys baselinecorretion from -0.2 to 0.0
        set M1 and M2 as reference 
        drops EOG channels
        '''
        eog_evoked = mne.preprocessing.create_eog_epochs(raw)
        eog_evoked.apply_baseline(baseline = (None, -0.5))
        ica = mne.preprocessing.ICA(n_components = 15, max_iter = 'auto', random_state = 42)
        ica.fit(raw)
        ica.exclude = []
        eog_incides, eog_scores = ica.find_bads_eog(raw)
        ica.exclude = eog_incides
        ica.apply(raw)
        raw.set_eeg_reference(ref_channels =  ['M1', 'M2'])
        raw.drop_channels(ch_names = ['VEOG', 'HEOG'])
        return raw


    def preprocess(self):
        '''
        function for precprcessing EEG data
        pereprocesses all data in of raw_dict with .set or .fif format
        applies filters as indicated
        '''        
        save_list = []
        files_for_preprocessing = self.get_datafiles()
        for index, row in files_for_preprocessing.iterrows():
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            raw = self.load_data(row['file_path'])
            vp = row['file_path'].split('/')[-2]

            # turn eog channels to eog in mne.info
            raw.set_channel_types({'VEOG': 'eog','HEOG': 'eog'})
            # apply monatge
            raw = raw.notch_filter(50)
            # montage = mne.channels.make_standard_montage('standard_1020')
            # raw.set_montage(montage)
            #raw = raw.resample(sfreq=raw.info['sfreq']/self.resample)
            #raw = self.apply_ica(raw)
            events_from_annot, event_dict = mne.events_from_annotations(raw)

            epochs = mne.Epochs(raw, events_from_annot, tmin = self.window_begin, tmax = self.window_end, event_id = event_dict, event_repeated = 'drop', preload = True, baseline = None)
                
            if '11' in event_dict:
                X_1, label_1 = self.epoch2data(epochs, '11')
                X_2, label_2 = self.epoch2data(epochs, '12')
                X_3, label_3 = self.epoch2data(epochs, '21')
                X_4, label_4 = self.epoch2data(epochs, '22')
                X_5, label_5 = self.epoch2data(epochs, '13')
                X_6, label_6 = self.epoch2data(epochs, '23')
            elif ' 11' in event_dict:
                X_1, label_1 = self.epoch2data(epochs, ' 11')
                X_2, label_2 = self.epoch2data(epochs, ' 12')
                X_3, label_3 = self.epoch2data(epochs, ' 21')
                X_4, label_4 = self.epoch2data(epochs, ' 22')
                X_5, label_5 = self.epoch2data(epochs, ' 13')
                X_6, label_6 = self.epoch2data(epochs, ' 23')
            else:
                print('erorr in:',event_dict)

            epochs = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis = 0 )
            label = np.concatenate([label_1, label_2, label_3, label_4, label_5, label_6], axis = 0)
            label = [item.strip() for item in label]

            # cheack and creates new folder for preprocessed data of participant
            if not os.path.exists(row['save_path']):
                os.makedirs(row['save_path'])

            save_path_X = row['save_path'] + vp + '_X_raw' + str(abs(self.window_begin)+self.window_end) +'s_ica_512Hz_NFEEG.npy'
            save_path_Y = row['save_path'] + vp + '_Y_raw' + str(abs(self.window_begin)+self.window_end) +'s_ica_512Hz_NFEEG.npy'
            save_list.append([int(vp), save_path_X, save_path_Y])
            np.save(save_path_X, epochs)
            np.save(save_path_Y, label)
        save_list = pd.DataFrame(save_list, columns=['vp' ,'save_path_X_NFEEG', 'save_path_Y_NFEEG'])
        save_list = save_list.sort_values(by=['vp'])
        return save_list
    

class FMIEEG_preprocessing(EEGNet_preprocessing):
    def __init__(self, high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample):
        super().__init__(high_pass, low_pass, window_begin, window_end, filterbank, raw_dict, resample)
        self.low_pass = filterbank[1]
        self.high_pass = filterbank[0]


    def preprocess(self):
        '''
        function for precprcessing EEG data
        pereprocesses all data in of raw_dict with .set or .fif format
        applies filters as indicated
        '''        
        save_list = []
        files_for_preprocessing = self.get_datafiles()
        for index, row in files_for_preprocessing.iterrows():
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            print(index, '#####################################################################################')
            raw = self.load_data(row['file_path'])
            vp = row['file_path'].split('/')[-2]

            # turn eog channels to eog in mne.info
            raw.set_channel_types({'VEOG': 'eog','HEOG': 'eog'})
            # apply monatge
            # montage = mne.channels.make_standard_montage('standard_1020')
            # raw.set_montage(montage)
            # raw = raw.resample(sfreq=raw.info['sfreq']/self.resample)
            #raw = self.apply_ica(raw)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            epochs_fb = []
            for i,_ in enumerate(self.high_pass):
                raw.filter(self.high_pass[i], self.low_pass[i])

                epochs = mne.Epochs(raw, events_from_annot, tmin = self.window_begin, tmax = self.window_end, event_id = event_dict, event_repeated = 'drop', preload = True, baseline = None)
                
                if '11' in event_dict:
                    X_1, label_1 = self.epoch2data(epochs, '11')
                    X_2, label_2 = self.epoch2data(epochs, '12')
                    X_3, label_3 = self.epoch2data(epochs, '21')
                    X_4, label_4 = self.epoch2data(epochs, '22')
                    X_5, label_5 = self.epoch2data(epochs, '13')
                    X_6, label_6 = self.epoch2data(epochs, '23')
                elif ' 11' in event_dict:
                    X_1, label_1 = self.epoch2data(epochs, ' 11')
                    X_2, label_2 = self.epoch2data(epochs, ' 12')
                    X_3, label_3 = self.epoch2data(epochs, ' 21')
                    X_4, label_4 = self.epoch2data(epochs, ' 22')
                    X_5, label_5 = self.epoch2data(epochs, ' 13')
                    X_6, label_6 = self.epoch2data(epochs, ' 23')
                else:
                    print('erorr in:',event_dict)

                epochs = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6], axis = 0 )
                epochs = np.expand_dims(epochs, axis = 1)
                label = np.concatenate([label_1, label_2, label_3, label_4, label_5, label_6], axis = 0)
                label = [item.strip() for item in label]
                #concat all filterbanks
                if len(epochs_fb) == 0:
                    epochs_fb = epochs
                else:
                    epochs_fb = np.concatenate((epochs_fb,epochs), axis = 1)

            epochs = epochs_fb

            # cheack and creates new folder for preprocessed data of participant
            if not os.path.exists(row['save_path']):
                os.makedirs(row['save_path'])

            save_path_X = row['save_path'] + vp + '_X_'+ str(self.high_pass[0]) + '-' + str(self.low_pass[-1]) + 'Hz-FB_' + str(abs(self.window_begin)+self.window_end) +'s_ica_512Hz_FMIEEG.npy'
            save_path_Y = row['save_path'] + vp + '_Y_'+ str(self.high_pass[0]) + '-' + str(self.low_pass[-1]) + 'Hz-FB_' + str(abs(self.window_begin)+self.window_end) +'s_ica_512Hz_FMIEEG.npy'
            save_list.append([int(vp), save_path_X, save_path_Y])
            np.save(save_path_X, epochs)
            np.save(save_path_Y, label)
        save_list = pd.DataFrame(save_list, columns=['vp' ,'save_path_X_FMIEEG', 'save_path_Y_FMIEEG'])
        save_list = save_list.sort_values(by=['vp'])
        return save_list
    

def data(root = '/home/administrator/Documents/Thesis/data/raw/'):
    ''''
    retuns a pd.dataframe of all locations where the data has been preprocessed
    if no save_list.xlsx then all preprocessing is done
    '''
    save = '/'.join(root.split('/')[:-2])+'/save_list.xlsx'
    
    if 'save_list.xlsx' not in os.listdir('/'.join(root.split('/')[:-2])):
        save_list1 = EEGNet_preprocessing(high_pass = 4.0, low_pass = 40, window_begin = -0.5, window_end = 4.0, filterbank = False, resample = 128, raw_dict = root).preprocess()
        save_list2 = FBCNet_preprocessing(high_pass = False, low_pass = False,window_begin=-0.49,window_end=4.0,filterbank=[[4,8,12,16,20,24,28,32,36],[8,12,16,20,24,28,32,36,40]],raw_dict=root,resample= 128).preprocess()
        save_list3 = NFEEG_preprocessing(high_pass = False, low_pass = False,window_begin=-0.5,window_end=4.0,filterbank=False,raw_dict=root,resample= 512).preprocess()
        save_list4 = FMIEEG_preprocessing(high_pass = False, low_pass = False,window_begin=-0.5,window_end=4.0,filterbank=[[1,4,8,13,31],[4,8,13,31,40]],raw_dict=root,resample= 512).preprocess()

        save_list = pd.concat([save_list1, save_list2, save_list3, save_list4], axis=1)
        save_list.to_excel(save)
        return save_list
    elif 'save_list.xlsx' in os.listdir('/'.join(root.split('/')[:-2])):
        return pd.read_excel(save)