import sys, tempfile, os, json
from types import SimpleNamespace
from run_point_hbf import build_argv
from run_sweep_b200 import hbm_kv_fraction
from vidur.execution_time_predictor.base_execution_time_predictor import BaseExecutionTimePredictor as B
CAP=[]; _o=B.get_execution_time
def _p(self,batch,stage):
    et=_o(self,batch,stage); CAP.append(et); return et
B.get_execution_time=_p
label,model,ctx,batch,tp,sp,tier,sra=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),sys.argv[7],sys.argv[8]=='1'
frac=hbm_kv_fraction(model,"blackwell",batch,ctx,tp,tier)
out=tempfile.mkdtemp();cache=tempfile.mkdtemp()
a=SimpleNamespace(model=model,device="blackwell",batch_size=batch,context_length=ctx,decode_tokens=4,
  num_requests=batch,qps=1000.0,tensor_parallel_size=tp,network_device="h100_dgx",sparsity_fraction=sp,
  hbm_kv_fraction=frac,hbf_sra=sra,naive_sparse=False,sparse_read_amplification=(3.5636 if sra else 1.0),
  plane_imbalance_factor=1.0,backing_bw_gbps=8000.0,num_kv_blocks=0,
  hbfsim_toml=os.path.abspath("configs/hbf_paper.toml"),cache_dir=cache)
sys.argv=build_argv(a,out)
from vidur.config import SimulationConfig; from vidur.simulator import Simulator; from vidur.utils.random import set_seeds
cfg=SimulationConfig.create_from_cli_args(); set_seeds(cfg.seed); Simulator(cfg).run()
# pick the FULLEST decode step (max attention_decode) -> full-batch, avoids partial
# chunks captured when high-batch steps are preempted/split.
et=max(CAP, key=lambda e: getattr(e,'_attention_decode_execution_time',0.0)); L=et.num_layers
g=lambda n: float(getattr(et,n,0.0) or 0.0)
comp=dict(
  kv_read=g("_attention_decode_execution_time")*L,
  attn_proj=(g("_attention_layer_pre_proj_execution_time")+g("_attention_layer_post_proj_execution_time")+g("_attention_rope_execution_time"))*L,
  mlp=(g("_mlp_layer_up_proj_execution_time")+g("_mlp_layer_down_proj_execution_time")+g("_mlp_layer_act_execution_time"))*L,
  kv_write=g("_attention_kv_cache_save_execution_time")*L,
  prefill=g("_attention_prefill_execution_time")*L,
  comm=g("_tensor_parallel_communication_time")*2*L,
  norms=(g("_attn_norm_time")+g("_mlp_norm_time")+g("_add_time"))*L,
  pp_comm=g("_pipeline_parallel_communication_time"),
  model_time_ms=et.model_time_ms,
  total_time_ms=et.total_time*1e3,
  block_total=et._get_block_execution_time()*L)
print("RESULT "+label+" "+json.dumps(comp))
