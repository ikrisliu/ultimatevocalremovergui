import os
import json
import shutil
import hashlib
import torch
from typing import List
from gui_data.constants import *
from separate import (
    SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR,  # Model-related
    save_format, clear_gpu_cache,  # Utility functions
    cuda_available, mps_available, #directml_available,
)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

model_hash_table = {}


def load_model_hash_data(dictionary):
    with open(dictionary, 'r') as d:
        return json.load(d)


class ModelData:
    def __init__(self, model_name: str,
                 selected_process_method=MDX_ARCH_TYPE,
                 is_secondary_model=False,
                 primary_model_primary_stem=None,
                 is_primary_model_primary_stem_only=False,
                 is_primary_model_secondary_stem_only=False,
                 is_pre_proc_model=False,
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False):

        self.model_path = ''
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = False
        self.deverb_vocal_opt = 'ALL'
        self.is_denoise_model = False
        self.is_gpu_conversion = 1
        self.is_normalization = False
        self.is_use_opencl = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_denoise = False
        self.is_mdx_c_seg_def = False
        self.mdx_batch_size = 1
        self.mdxnet_stem_select = ALL_STEMS
        self.overlap = 0.25
        self.overlap_mdx = DEFAULT
        self.overlap_mdx23 = 8
        self.semitone_shift = 0
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = True
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = False
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = 'PCM_16'
        self.device_set = '0'
        self.mp3_bit_set = '320k'
        self.save_format = 'WAV'
        self.is_invert_spec = False
        self.is_mixer_mode = False
        self.demucs_stems = ALL_STEMS
        self.is_demucs_combine_stems = False
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = False
        self.is_inst_only_voc_splitter = False
        self.is_save_vocal_only = False
        self.mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = load_model_hash_data(MDX_MODEL_NAME_SELECT)

        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = False
            self.margin = 44100
            self.chunks = 0
            self.mdx_segment_size = 256
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR, self.mdx_hash_MAPPER)
                if self.model_data:

                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        # config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        # if os.path.isfile(config_path):
                        #     with open(config_path) as f:
                        #         config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                        #
                        #     self.mdx_c_configs = config
                        #
                        #     if self.mdx_c_configs.training.target_instrument:
                        #         # Use target_instrument as the primary stem and set 4-stem ensemble to False
                        #         target = self.mdx_c_configs.training.target_instrument
                        #         self.mdx_model_stems = [target]
                        #         self.primary_stem = target
                        #     else:
                        #         # If no specific target_instrument, use all instruments in the training config
                        #         self.mdx_model_stems = self.mdx_c_configs.training.instruments
                        #         self.mdx_stem_count = len(self.mdx_model_stems)
                        #
                        #         # Set primary stem based on stem count
                        #         if self.mdx_stem_count == 2:
                        #             self.primary_stem = self.mdx_model_stems[0]
                        #         else:
                        #             self.primary_stem = self.mdxnet_stem_select
                        #
                        #         # Update mdxnet_stem_select based on ensemble mode
                        #         if self.is_ensemble_mode:
                        #             self.mdxnet_stem_select = self.ensemble_primary_stem
                        # else:
                        #     self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"]
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()

                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None

        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False

        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = True
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)

                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0

                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = BV_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else LEAD_VOCAL_STEM
            else:
                primary = LEAD_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else BV_VOCAL_STEM
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)

        self.vocal_splitter_model_data()

    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = None
            self.is_vocal_split_model_activated = True if self.vocal_split_model else False

            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True

    def secondary_model_data(self, primary_stem):
        secondary_model_data = None
        # self.secondary_model = secondary_model_data[0]
        # self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True

    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]

    def get_mdx_model_path(self):
        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX

        for file_name, chosen_mdx_model in self.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")

        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")

    def get_model_data(self, model_hash_dir, hash_mapper:dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

            return None

    def change_model_data(self):
        if self.is_get_hash_dir_only:
            return None
        else:
            return None

    def get_model_hash(self):
        self.model_hash = None

        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break

            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()

                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)


class UVR:
    def __init__(self, input_paths: [str], export_path: str):
        self.true_model_count = 0
        self.iteration = 0
        self.input_paths = input_paths
        self.export_path = export_path
        self.is_ensemble = False
        self.mdx_primary_model_names = []
        self.mdx_secondary_model_names = []
        self.demucs_secondary_model_names = []
        self.all_models = []
        self.mdx_cache_source_mapper = {}
        self.clear_cache_torch = False

    def cached_source_model_list_check(self, model_list: List[ModelData]):
        model: ModelData
        primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
        secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

        self.mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
        self.mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)

        for model in model_list:
            if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
                if not model.is_4_stem_ensemble:
                    self.demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                    break
                else:
                    for i in model.secondary_model_4_stem_model_names_list:
                        self.demucs_secondary_model_names.append(i)

        self.all_models = self.mdx_primary_model_names + self.mdx_secondary_model_names

    def determine_voc_split(self, models):
        # is_vocal_active = self.check_only_selection_stem(VOCAL_STEM_ONLY) or self.check_only_selection_stem(INST_STEM_ONLY)
        #
        # if self.set_vocal_splitter_var.get() != NO_MODEL and self.is_set_vocal_splitter_var.get() and is_vocal_active:
        #     model_stems_list = self.model_list(VOCAL_STEM, INST_STEM, is_dry_check=True, is_check_vocal_split=True)
        #     if any(model.model_basename in model_stems_list for model in models):
        #         return 1
        return 0

    def cached_sources_clear(self):
        self.mdx_cache_source_mapper = {}

    def process_iteration(self):
        self.iteration = self.iteration + 1

    def cached_source_callback(self, process_method, model_name=None):
        model, sources = None, None

        if process_method == MDX_ARCH_TYPE:
            mapper = self.mdx_cache_source_mapper

        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value

        return model, sources

    def cached_model_source_holder(self, process_method, sources, model_name=None):
        if process_method == MDX_ARCH_TYPE:
            self.mdx_cache_source_mapper = {**self.mdx_cache_source_mapper, **{model_name: sources}}

    def print_gpu_list(self):
        try:
            if cuda_available:
                self.cuda_device_list = [f"{torch.cuda.get_device_properties(i).name}:{i}" for i in range(torch.cuda.device_count())]
                self.cuda_device_list.insert(0, DEFAULT)
                print(self.cuda_device_list)
        except Exception as e:
            print(e)

    def process_start(self):
        self.print_gpu_list()
        try:
            model = [ModelData('UVR-MDX-NET-Inst_HQ_3', MDX_ARCH_TYPE)]
            self.cached_source_model_list_check(model)

            true_model_4_stem_count = sum(0 for m in model)
            true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
            self.true_model_count = (sum(2 if m.is_secondary_model_activated else 1 for m in model) +
                                     true_model_4_stem_count + true_model_pre_proc_model_count +
                                     self.determine_voc_split(model))

            for file_num, audio_file in enumerate(self.input_paths, start=1):
                self.cached_sources_clear()

                for current_model_num, current_model in enumerate(model, start=1):
                    self.iteration += 1

                    process_data = {
                        'model_data': current_model,
                        'export_path': self.export_path,
                        'audio_file_base': f"{file_num}_{os.path.splitext(os.path.basename(audio_file))[0]}",
                        'audio_file': audio_file,
                        'set_progress_bar': None,
                        'write_to_console': None,
                        'process_iteration': self.process_iteration,
                        'cached_source_callback': self.cached_source_callback,
                        'cached_model_source_holder': self.cached_model_source_holder,
                        'list_all_models': self.all_models,
                        'is_ensemble_master': self.is_ensemble,
                        'is_4_stem_ensemble': False
                    }

                    seperator = SeperateMDX(current_model, process_data)
                    seperator.seperate()

                clear_gpu_cache()

            shutil.rmtree(self.export_path) if self.is_ensemble and len(os.listdir(self.export_path)) == 0 else None

            self.process_end()

        except Exception as ex:
            self.process_end(error=ex)
            raise ex

    def process_end(self, error=None):
        self.cached_sources_clear()
        self.clear_cache_torch = True
