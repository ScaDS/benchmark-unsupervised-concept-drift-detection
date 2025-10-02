export PATH="/data/horse/ws/s4122485-compPerfDD/benchmark/dfki/benchmarkdd/OmniOpt/":$PATH
export DONT_SEND_SIGNAL_BEFORE_END=1
# Define parameters for different DDs, parameter_name:dtype:min:max
CSDDM=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "confidence:choice:float:0.25,0.1,0.05,0.025,0.01,0.005,0.001" "feature_proportion:range:float:0.001 0.999" "n_clusters:range:int:1 50")
BNDM=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "const:range:float:0.001 10.0" "max_depth:range:int:1 20" "threshold:range:float:0.001 0.999")
D3=("recent_samples_size:range:int:1 5000" "n_reference_samples:range:int:1 5000" "recent_samples_proportion:range:float:0.001 0.999" "threshold:range:float:0.001 0.999")
# IBDD update interval should be gt 1, otherwise error, n permutations as well
IBDD=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "n_permutations:range:int:2 1000" "update_interval:range:int:2 250" "n_consecutive_deviations:range:int:1 250")
OCDD=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "threshold:range:float:0.001 0.999" "outlier_detector_kwargs:fixed:_:_:\{\'nu\':0.5,\'kernel\':\'rbf\',\'gamma\':\'auto\'\}")
SPLL=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "n_clusters:range:int:1 10" "threshold:range:float:0.001 0.999")
UDetect=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "n_windows:range:int:1 5000" "disjoint_training_windows:choice:bool:True,False")
# EDFS didnt implement the EDFSMode.SUBSPACE_SELECTION, we should check the paper for re-implementation
EDFS=("recent_samples_size:range:int:1 5000" "n_subspaces:range:int:1 100" "feature_percentage:range:float:0.001 0.999" "alpha:range:float:0.001 0.999" "window_size:range:int:100 1000" "mode:choice:enum:EDFSMode.RANDOM")
#MD3 has an additional "clf" and a "margin_calculation_function" parameter
#MD3=("recent_samples_size:range:int:1 5000" "oracle_data_length_required:range:int:1 5000" "k:range:int:2 50" "sensitivity:range:float:0.5 2.0")
NNDVI=("recent_samples_size:range:int:1 5000" "n_samples:range:int:1 5000" "k_neighbors:range:int:1 100" "n_permutations:range:int:1 1000" "significance_level:range:float:0.001 0.999")
UCDD=("recent_samples_size:range:int:1 5000" "n_reference_samples:range:int:1 5000" "n_recent_samples:range:int:1 5000" "threshold:range:float:0.001 0.999" "stability_offset:choice:float:0.000000000001,0.00000000001,0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
STUDD=("recent_samples_size:range:int:1 5000" "delta:range:float:0.001 0.999")
DDAL=("recent_samples_size:range:int:1 5000" "batch_size:range:int:1 5000" "theta:range:float:0.0001 0.999" "lambida:range:float:0.001 0.999")
IKS=("recent_samples_size:range:int:1 5000" "feature_id:range:int:0 0" "threshold:range:float:0.001 0.999" "window_size:range:int:1 5000")
DAWIDD=("recent_samples_size:range:int:1 5000" "max_window_size:range:int:1 5000" "min_window_size:range:int:1 5000" "min_p_value:range:float:0.0001 0.999")
# for HDDDM it configurations where mmd2=True and k2s_test=False are invalid, yet not good way to filter them, batch size 1 leads to devision by zero exception
HDDDM=("recent_samples_size:range:int:1 5000" "batch_size:range:int:2 5000" "gamma:range:float:0.001 10.0" "alpha:range:float:0.0005 0.05" "use_mmd2:choice:bool:True,False" "use_k2s_test:choice:bool:True,False")
ETFE=("recent_samples_size:range:int:1 5000" "window_len:range:int:1 5000" "threshold:range:int:1 10" "startup:range:int:100 2500" "H:range:int:50 250" "entropy_type:choice:str:str\(\'PeEn\'\),str\(\'WPeEn\'\),str\(\'FsEn\'\),str\(\'IncrEn\'\),str\(\'ApEn\'\),str\(\'SampEn\'\)")
#LD3=("recent_samples_size:range:int:1 5000" "k:range:int:1 10" "window_size:range:int:100 1000" "detection_window_size:range:int:1 100" "label_count:range:int:1000 50000" "big_dataset:choice:bool:True,False")
# PCACD window size 1 would lead to exception
PCACD=("recent_samples_size:range:int:1 5000" "window_size:range:int:2 5000" "ev_threshold:range:float:0.001 0.999" "delta:range:float:0.001 0.999" "divergence_metric:choice:str:str\(\'kl\'\),str\(\'intersection\'\)" "sample_period:range:float:0.01 0.1" "online_scaling:choice:bool:True,False")
CDBD=("recent_samples_size:range:int:1 5000" "feature_id:range:int:0 0" "batch_size:range:int:1 5000" "divergence:choice:str:str\(\'KL\'\),str\(\'H\'\)" "detect_batch:choice:int:1,2,3" "statistic:choice:str:str\(\'tstat\'\),str\(\'stdev\'\)" "significance:range:float:0.001 0.999" "subsets:range:int:1 100")
MCDDD=("recent_samples_size:range:int:1 5000" "epochs:range:int:1 500" "sub_window_num:range:int:1 500" "n:range:int:1 1000" "k:choice:int:2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100" "eps_small:range:int:1 500" "eps_big:range:int:1 500" "temperature:range:float:0.01 0.999" "lamb:range:int:1 500" "percentile:range:float:0.001 0.999")
# for WindowKDE, n_training_samples >= recent_samples_size >= big_windowSize >= small_windowSize
WindowKDE=("recent_samples_size:range:int:1 5000" "feature_id:range:int:0 0" "p:range:int:1 500" "big_windowSize:range:int:2 1999" "small_windowSize:range:int:2 1999")
# original slid shaps: SlidShaps=("recent_samples_size:range:int:1 5000" "batch_size:range:int:1 5000" "detection_buf_size:range:int:1 5000" "overlap:range:float:0.001 0.999" "alpha:range:float:0.001 0.999" "gamma:range:float:0.001 10" "statistical_test:choice:str:str\(\'t-test\'\),str\(\'ks-test\'\)" "approximation_type:choice:str:str\(\'full\'\),str\(\'bounded\'\)" "subset_bound:range:int:1 5000")
SlidShaps=("recent_samples_size:range:int:1 5000" "batch_size:range:int:1 5000" "detection_buf_size:range:int:1 5000" "overlap:range:float:0.001 0.999" "alpha:range:float:0.001 0.999" "gamma:range:float:0.001 10" "statistical_test:choice:str:str\(\'t-test\'\),str\(\'ks-test\'\)" "approximation_type:choice:str:str\(\'bounded\'\)" "subset_bound:range:int:1 50")
CDLEEDS=("recent_samples_size:range:int:1 5000" "significance:range:float:0.001 0.999" "gamma:range:float:0.001 0.999" "max_node_size:range:int:1 5000" "max_tree_depth:range:int:1 3500" "max_time_stationary:range:int:1 5000")
NNBDD=("recent_samples_size:range:int:1 5000" "batch_size:range:int:4 512" "threshold:range:float:0.001 0.999" "epochs:range:int:1 500")

# define number of features per dataset, required for univariat DDs
# always -1 since we use it as index, starting with 0
TMDBalanced5s=39
Electricity=7
NOAAWeather=7
OutdoorObjects=20
PokerHand=9
Powersupply=1
RialtoBridgeTimelapse=26
ForestCovertype=53
GasSensor=127
Keystroke=9
Luxembourg=30
Ozone=71
SensorStream=4
SineClusters=4

# synthetic streams
WaveformPre=21
SineClustersPre=4

n_training_samples=2000

constraint_PCACD=("sample_period*window_size >= 1")
constraint_NNDVI=("2*n_samples >= k_neighbors+1")
constraint_SlidShaps=("overlap*batch_size >= 1")
constraint_IBDD=("n_permutations >= n_consecutive_deviations" "update_interval >= n_consecutive_deviations+1")
constraint_WindowKDE=("small_windowSize+1 <= big_windowSize+1" "big_windowSize+1 <= recent_samples_size" "big_windowSize >= small_windowSize*4")
constraint_MCDDD=("eps_small+1 <= eps_big")
# we only have to define it for CDLEEDS, since changed parameter sets are not considered yet
constraint_CDLEEDS=("max_tree_depth <= 3500")
