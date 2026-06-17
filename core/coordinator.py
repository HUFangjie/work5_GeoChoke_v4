import csv, os, importlib, numpy as np
from config import CONFIG
from utils.seed import set_seed
from utils.logger import setup_logger
from data.mnist import MNISTProvider
import models.mnist_cnn
from models.registry import create_model
from crypto.ckks_context_manager import CKKSContextManager
from crypto.ckks_backend import CKKSBackend
from crypto.update_codec import ModelUpdateCodec
from core.decryption_service import AuthorizedDecryptionService
from core.server import AggregationServer
from core.client import Client
from attacks.no_attack import NoAttack
from attacks.alie import ALIEAttack
from attacks.fang import FangMeanAttack
from defenses.geochoke.defense import GeoChokeDefense
from evaluation.evaluator import Evaluator
from utils.validation import model_l2_norm

def build_attack(cfg):
    if cfg.attack_type=='alie': return ALIEAttack(cfg.alie_z, oracle_all_updates=cfg.alie_oracle_all_updates)
    if cfg.attack_type=='fang_mean': return FangMeanAttack(cfg.aggregation,cfg.fang_max_norm,cfg.fang_search_steps)
    return NoAttack()

def run(cfg=CONFIG):
    set_seed(cfg.seed); log=setup_logger(cfg.log_level); os.makedirs(cfg.output_dir,exist_ok=True)
    splits=MNISTProvider(cfg).build(); model=create_model(cfg.model_name); codec=ModelUpdateCodec(model)
    cm=CKKSContextManager(cfg.ckks_profiles); backend=CKKSBackend(cm,cfg.ckks_profiles); backend.initialize_profiles(); dec=AuthorizedDecryptionService(cm)
    defense=GeoChokeDefense(cfg.geochoke,cfg.ckks_profiles); defense.initialize(model,backend,splits.proxy_loader)
    clients=[]; attack=build_attack(cfg)
    for cid,loader in enumerate(splits.client_loaders): clients.append(Client(cid,loader,lambda:create_model(cfg.model_name),lambda m:ModelUpdateCodec(m),backend,attack,cid in cfg.malicious_client_ids,cfg))
    server=AggregationServer(cfg,model,codec,backend,dec,defense); evaluator=Evaluator(splits.test_loader,cfg.device)
    rows=[]
    for r in range(cfg.num_rounds):
        pid=defense.get_profile_for_round(r); selected=server.sample_clients(r); log.info('round %s profile %s clients %s',r,pid,selected)
        global_state={k:v.detach().cpu().clone() for k,v in server.model.state_dict().items()}
        # benign observable context from clean local updates is approximated by each malicious client's own clean update unless oracle metrics enabled.
        uploads=[]
        ctx={'num_selected':len(selected),'num_malicious':len([i for i in selected if i in cfg.malicious_client_ids]),'observable_updates':[]}
        for cid in selected: uploads.append(clients[cid].run_round(global_state,pid,ctx))
        dec_update,metrics=server.apply_round(uploads,r); test_loss,test_acc=evaluator.evaluate(server.model)
        row={'round':r,'selected_clients':selected,'malicious_selected_clients':[i for i in selected if i in cfg.malicious_client_ids],'train_loss':np.mean([u.metadata['train_loss'] for u in uploads]),'test_loss':test_loss,'test_accuracy':test_acc,'global_model_norm':model_l2_norm(server.model),'profile_id':pid,'poly_modulus_degree':cfg.ckks_profiles[pid]['poly_modulus_degree'],'scale_bits':cfg.ckks_profiles[pid]['global_scale_bits'],'coeff_modulus_bits':cfg.ckks_profiles[pid]['coeff_mod_bit_sizes'],'encryption_time':sum(backend.last_encryption_time for _ in uploads),'attack_type':cfg.attack_type,**metrics}
        rows.append(row)
        with open(os.path.join(cfg.output_dir,f'round_{r}.csv'),'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    return rows
