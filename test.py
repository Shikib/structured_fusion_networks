import argparse
import json
import math
import model
import random
import subprocess
import time

from collections import defaultdict, Counter
#from evaluate_model import evaluateModel

import numpy as np
from sklearn.metrics import f1_score

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='MultiWoz Training Script')

parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--use_attn', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--test_only', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--model_name', type=str, default='baseline')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--domain', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--test_lm', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--concat_da', type=str2bool, default=False)

parser.add_argument('--emb_size', type=int, default=50)
parser.add_argument('--hid_size', type=int, default=150)
parser.add_argument('--db_size', type=int, default=30)
parser.add_argument('--bs_size', type=int, default=94)
parser.add_argument('--da_size', type=int, default=593)

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

parser.add_argument('--shallow_fusion', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--deep_fusion', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--cold_fusion', type=str2bool, const=True, nargs='?', default=False)

parser.add_argument('--bs_predictor', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--dm_predictor', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--nlg_predictor', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--s2s_predictor', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--naive_fusion', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--multitask_model', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--structured_fusion', type=str2bool, const=True, nargs='?', default=False)



parser.add_argument('--lm_name', type=str, default='baseline')
parser.add_argument('--s2s_name', type=str, default='baseline')


args = parser.parse_args()

def load_data(filename, dial_acts_data, dial_act_dict):
  data = json.load(open(filename))
  rows = []
  for file, dial in data.items():
    input_so_far = []
    for i in range(len(dial['sys'])):
      input_so_far += ['_GO'] + dial['usr'][i].strip().split() + ['_EOS']

      input_seq = [e for e in input_so_far]
      target_seq = ['_GO'] + dial['sys'][i].strip().split() + ['_EOS']
      db = dial['db'][i]
      bs = dial['bs'][i]

      # Get dialog acts
      dial_turns = dial_acts_data[file.strip('.json')]
      if str(i+1) not in dial_turns.keys():
        da = [0.0]*len(dial_act_dict)
      else:
        turn = dial_turns[str(i+1)]
        da = [0.0]*len(dial_act_dict)
        if turn != "No Annotation":
          for act_type, slots in turn.items():
            domain = act_type.split("-")[0]
            da[dial_act_dict["d:"+domain]] = 1.0
            da[dial_act_dict["d-a:"+act_type]] = 1.0
            for slot in slots:
              dasv = "d-a-s-v:" + act_type + "-" + slot[0] + "-" + slot[1]
              da[dial_act_dict[dasv]] = 1.0

      rows.append((input_seq, target_seq, db, bs, da, file, i))

  return rows

#def load_data(filename, dial_acts_data, dial_act_dict):
#  data = json.load(open(filename))
#  rows = []
#  for file, dial in data.items():
#    input_so_far = []
#    for i in range(len(dial['sys'])):
#      input_so_far += ['_GO'] + dial['usr'][i].strip().split() + ['_EOS']
#
#      input_seq = [e for e in input_so_far]
#      target_seq = ['_GO'] + dial['sys'][i].strip().split() + ['_EOS']
#      db = dial['db'][i]
#      bs = dial['bs'][i]
#
#      # Get dialog acts
#      dial_turns = dial_acts_data[file.strip('.json')]
#      if str(i+1) not in dial_turns.keys():
#        da = [0.0]*len(dial_act_dict)
#      else:
#        turn = dial_turns[str(i+1)]
#        da = [0.0]*len(dial_act_dict)
#        if turn != "No Annotation":
#          for act in turn.keys():
#            da[dial_act_dict[act]] = 1.0
#
#      rows.append((input_seq, target_seq, db, bs, da, file, i))
#
#      # Add sys output
#      input_so_far += target_seq
#
#  return rows
#
#
#def get_dial_acts(filename):
#
#  data = json.load(open(filename))
#  dial_acts = []
#  for dial in data.values():
#    for turn in dial.values():
#      if turn == "No Annotation":
#        continue
#      for dial_act in turn.keys():
#        if dial_act not in dial_acts:
#          dial_acts.append(dial_act)
#  print(dial_acts, len(dial_acts))
#  return dict(zip(dial_acts, range(len(dial_acts)))), data



def get_dial_acts(filename):

  dial_acts_data = json.load(open(filename))

  with open('data/template.txt') as f:
    dial_acts = [x.strip('\n') for x in f.readlines()]

  return dict(zip(dial_acts, range(len(dial_acts)))), dial_acts_data
  # data = json.load(open(filename))
  # dial_acts = []
  # for dial in data.values():
  #   for turn in dial.values():
  #     if turn == "No Annotation":
  #       continue
  #     for dial_act_type, slots in turn.items():
  #       for slot in slots:
  #         dial_act = dial_act_type + "_" + slot[0]
  #         if dial_act not in dial_acts:
  #           dial_acts.append(dial_act)
  # print(dial_acts, len(dial_acts))
  # return dict(zip(dial_acts, range(len(dial_acts)))), data

def get_belief_state_domains(bs):
  doms = []
  if 1 in bs[:14]:
    doms.append(u'taxi')
  if 1 in bs[14:31]:
    doms.append(u'restaurant')
  if 1 in bs[31:36]:
    doms.append(u'hospital')
  if 1 in bs[36:62]:
    doms.append(u'hotel')
  if 1 in bs[62:73]:
    doms.append(u'attraction')
  if 1 in bs[73:92]:
    doms.append(u'train')
  if 1 in bs[92:]:
    doms.append(u'police')
  return doms

def get_dialogue_domains(dial):

  doms = []
  for i in range(len(dial['sys'])):
    bs_doms = get_belief_state_domains(dial['bs'][i])
    doms = list(set(doms).union(bs_doms))
  return doms

def load_domain_data(filename, domains, include=False):

  data = json.load(open(filename))
  rows = []
  num_dials = 0
  num_total_dials = 0
  for filename, dial in data.items():
    input_so_far = []
    this_dial = False
    num_total_dials += 1
    for i in range(len(dial['sys'])):
      bs_doms = get_belief_state_domains(dial['bs'][i])
      if include:
        # Skip utterances which do not have any domains in 'domains'
        if len(list(set(bs_doms).intersection(domains))) == 0:
          continue
      else:
        # Skip utterances which have one of the domains in 'domains'
        if len(list(set(bs_doms).intersection(domains))) > 0:
          continue
      if this_dial is False:
        num_dials += 1
        this_dial = True
      input_so_far += ['_GO'] + dial['usr'][i].strip().split() + ['_EOS']
      input_seq = [e for e in input_so_far]
      target_seq = ['_GO'] + dial['sys'][i].strip().split() + ['_EOS']
      db = dial['db'][i]
      bs = dial['bs'][i]

      # Get dialog acts
      dial_turns = dial_acts_data[filename.strip('.json')]
      if str(i+1) not in dial_turns.keys():
        da = [0.0]*len(dial_act_dict)
      else:
        turn = dial_turns[str(i+1)]
        da = [0.0]*len(dial_act_dict)
        if turn != "No Annotation":
          for act_type, slots in turn.items():
            domain = act_type.split("-")[0]
            da[dial_act_dict["d:"+domain]] = 1.0
            da[dial_act_dict["d-a:"+act_type]] = 1.0
            for slot in slots:
              dasv = "d-a-s-v:" + act_type + "-" + slot[0] + "-" + slot[1]
              da[dial_act_dict[dasv]] = 1.0


      rows.append((input_seq, target_seq, db, bs, da, filename, i))

      # Add sys output
      input_so_far += target_seq

  print("Number of dialogues with restaurant in domains:", num_dials, "/", num_total_dials)
  return rows  

# Load vocabulary
input_w2i = json.load(open('data/input_lang.word2index.json'))
output_w2i = json.load(open('data/output_lang.word2index.json'))
input_i2w = json.load(open('data/input_lang.index2word.json'))
output_i2w = json.load(open('data/output_lang.index2word.json'))


dial_act_dict, dial_acts_data = get_dial_acts('data/dialogue_act_feats.json')

# Create models
encoder = model.Encoder(vocab_size=len(input_w2i), 
                                    emb_size=args.emb_size, 
                                    hid_size=args.hid_size)

policy = model.Policy(hidden_size=args.hid_size,
                      db_size=args.db_size,
                      bs_size=args.bs_size,
                      da_size=args.da_size)

decoder = model.Decoder(emb_size=args.emb_size,
                        hid_size=args.hid_size,
                        vocab_size=len(output_w2i),
                        use_attn=args.use_attn)

if args.shallow_fusion or args.deep_fusion:
  s2s = model.Model(encoder=encoder,
                    policy=policy,
                    decoder=decoder,
                    input_w2i=input_w2i,
                    output_w2i=output_w2i,
                    args=args)
  lm_decoder = model.Decoder(emb_size=args.emb_size,
                             hid_size=args.hid_size,
                             vocab_size=len(output_w2i),
                             use_attn=False)
  lm = model.LanguageModel(decoder=lm_decoder,
                           input_w2i=input_w2i,
                           output_w2i=output_w2i,
                           args=args)
  if args.shallow_fusion:
    model = model.ShallowFusionModel(s2s, lm, args)
  elif args.deep_fusion:
    model = model.DeepFusionModel(s2s, lm, args)
elif args.cold_fusion:
  s2s = model.Model(encoder=encoder,
                    policy=policy,
                    decoder=decoder,
                    input_w2i=input_w2i,
                    output_w2i=output_w2i,
                    args=args)
  lm_decoder = model.Decoder(emb_size=args.emb_size,
                             hid_size=args.hid_size,
                             vocab_size=len(output_w2i),
                             use_attn=False)
  lm = model.LanguageModel(decoder=lm_decoder,
                           input_w2i=input_w2i,
                           output_w2i=output_w2i,
                           args=args)
  cf = model.ColdFusionLayer(hid_size=args.hid_size,
                             vocab_size=len(output_w2i))
  model = model.ColdFusionModel(s2s, lm, cf, args)
elif args.bs_predictor:
  encoder = model.Encoder(vocab_size=len(input_w2i),
                          emb_size=args.emb_size,
                          hid_size=args.hid_size)
  model = model.NLU(encoder=encoder,
                    input_w2i=input_w2i,
                    args=args)
elif args.dm_predictor:
  pnn = model.PolicySmall(hidden_size=args.hid_size,
                          db_size=args.db_size,
                          bs_size=args.bs_size,
                          da_size=args.da_size)
  model = model.DM(pnn=pnn,
                   args=args)
elif args.nlg_predictor:
  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn,
                          concat_da=args.concat_da)
  model = model.NLG(decoder=decoder,
                    output_w2i=output_w2i,
                    args=args)
elif args.s2s_predictor:
  model = model.Model(encoder=encoder,
                      policy=policy,
                      decoder=decoder,
                      input_w2i=input_w2i,
                      output_w2i=output_w2i,
                      args=args).cuda()

 
elif args.naive_fusion:
  # Base components
  encoder = model.Encoder(vocab_size=len(input_w2i),
                          emb_size=args.emb_size,
                          hid_size=args.hid_size)
  nlu = model.NLU(encoder=encoder,
                    input_w2i=input_w2i,
                    args=args)

  pnn = model.PolicySmall(hidden_size=args.hid_size,
                          db_size=args.db_size,
                          bs_size=args.bs_size,
                          da_size=args.da_size)
  dm = model.DM(pnn=pnn,
                args=args)

  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  nlg = model.NLG(decoder=decoder,
                    output_w2i=output_w2i,
                    args=args)

  model = model.NaiveFusion(nlu=nlu,
                            dm=dm,
                            nlg=nlg,
                            input_w2i=input_w2i,
                            output_w2i=output_w2i,
                            args=args).cuda()

elif args.multitask_model:

  # Base components
  encoder = model.Encoder(vocab_size=len(input_w2i),
                          emb_size=args.emb_size,
                          hid_size=args.hid_size)
  nlu = model.NLU(encoder=encoder,
                    input_w2i=input_w2i,
                    args=args)

  pnn = model.PolicyBig(hidden_size=args.hid_size,
                          db_size=args.db_size,
                          bs_size=args.bs_size,
                          da_size=args.da_size)
  dm = model.MultiTaskedDM(pnn=pnn,
                args=args)


  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  nlg = model.NLG(decoder=decoder,
                    output_w2i=output_w2i,
                    args=args)

  e2e = model.E2E(encoder=encoder,
                    pnn=pnn,
                    decoder=decoder,
                    input_w2i=input_w2i,
                    output_w2i=output_w2i,
                    args=args).cuda()

  model = model.MultiTask(nlu=nlu,
                          dm=dm,
                          nlg=nlg,
                          e2e=e2e,
                          args=args).cuda()


elif args.structured_fusion:

  # NLU
  nlu_encoder = model.Encoder(vocab_size=len(input_w2i), 
                              emb_size=args.emb_size, 
                              hid_size=args.hid_size)
  nlu = model.NLU(encoder=nlu_encoder,
                  input_w2i=input_w2i,
                  args=args).cuda()
  
  # DM
  dm_pnn = model.PolicySmall(hidden_size=args.hid_size,
                             db_size=args.db_size,
                             bs_size=args.bs_size,
                             da_size=args.da_size)
  dm = model.DM(pnn=dm_pnn,
                args=args).cuda()

  # NLG
  nlg_decoder = model.Decoder(emb_size=args.emb_size,
                              hid_size=args.hid_size,
                              vocab_size=len(output_w2i),
                              use_attn=args.use_attn)
  nlg= model.NLG(decoder=nlg_decoder,
                 output_w2i=output_w2i,
                 args=args) 


  # Full model
  encoder = model.FusionEncoder(vocab_size=len(input_w2i), 
                                emb_size=args.emb_size, 
                                bs_size=args.bs_size, 
                                hid_size=args.hid_size)
  pnn = model.FusionPolicy(hidden_size=args.hid_size,
                           db_size=args.db_size,
                           bs_size=args.bs_size,
                           da_size=args.da_size)
  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  cf_dec = model.ColdFusionLayer(hid_size=args.hid_size,
                                 vocab_size=len(output_w2i))
  model = model.StructuredFusion(nlu=nlu,
                                dm=dm,
                                nlg=nlg,
                                encoder=encoder,
                                pnn=pnn,
                                cf_dec=cf_dec,
                                decoder=decoder,
                                input_w2i=input_w2i,
                                output_w2i=output_w2i,
                                args=args).cuda()



elif not args.test_lm:
  encoder = model.Encoder(vocab_size=len(input_w2i),
                          emb_size=args.emb_size,
                          hid_size=args.hid_size)

  pnn = model.PNN(hidden_size=args.hid_size,
                  db_size=args.db_size)

  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  #model  = model.E2E(encoder=encoder,
  #                  pnn=pnn,
  #                  decoder=decoder,
  #                  input_w2i=input_w2i,
  #                  output_w2i=output_w2i,
  #                  args=args).cuda()

  model = model.Model(encoder=encoder,
                      policy=policy,
                      decoder=decoder,
                      input_w2i=input_w2i,
                      output_w2i=output_w2i,
                      args=args)
else:
  model = model.LanguageModel(decoder=decoder,
                              input_w2i=input_w2i,
                              output_w2i=output_w2i,
                              args=args)
if args.use_cuda is True:
  model = model.cuda()

# Load data
train = load_data('data/train_dials.json', dial_acts_data, dial_act_dict)
valid = load_data('data/val_dials.json', dial_acts_data, dial_act_dict)
test = load_data('data/test_dials.json', dial_acts_data, dial_act_dict)

# Load domain data
if args.domain:
  test_domains = [u'restaurant']
  train = load_domain_data('data/train_dials.json', test_domains, include=False)
  valid = load_domain_data('data/val_dials.json', test_domains, include=False)
  test = load_domain_data('data/test_dials.json', test_domains, include=False)

num_val_batches = math.ceil(len(valid)/args.batch_size)

indices = list(range(len(test)))

model_name = args.model_name

best_val_score = 0.0
best_val_epoch = -1

if args.domain:
  data = json.load(open('data/val_dials.json'))
  fname_to_indices = defaultdict(list)
  for e in valid:
    fname_to_indices[e[-2]].append(e[-1])
  
  new_data = {}
  for fname,inds in fname_to_indices.items():
    new_data[fname] = {k:[v[i] for i in inds] for k,v in data[fname].items()}
  
  json.dump(new_data, open('temp_true.json', 'w+'))

for epoch in range(20 if not args.test_only else 0):
  indices = list(range(len(valid)))
  # Load saved model parameters
  model.load(model_name+"_"+str(epoch))
  #lm.load('id_lm_10')
  #model.lm = lm.decoder
  all_predicted = defaultdict(list)
  bs_predictions = []
  for batch in range(num_val_batches):
    # Prepare batch
    batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
    batch_rows = [valid[i] for i in batch_indices]

    if args.bs_predictor:
      input_seq, input_lens, bs = model.prep_batch(batch_rows)
      predicted_bs = model.predict(input_seq, input_lens)
      bs_predictions.append((predicted_bs.data.cpu().numpy(), bs.data.cpu().numpy()))
    elif args.dm_predictor:
      bs, da, db = model.prep_batch(batch_rows)
      predicted_da = model.predict(bs, db)
      bs_predictions.append((predicted_da.data.cpu().numpy(), da.data.cpu().numpy()))
    elif args.nlg_predictor:
      target_seq, target_lens, db, da, bs = model.prep_batch(batch_rows)
      # Get predicted sentences for batch
      predicted_sentences = model.decode(50, db, da, bs)

      # Add predicted to list
      for i,sent in enumerate(predicted_sentences):
        all_predicted[batch_rows[i][-2]].append(sent) 
    else:
      input_seq, input_lens, target_seq, target_lens, db, bs, da = model.prep_batch(batch_rows)

      # Get predicted sentences for batch
      predicted_sentences = model.decode(input_seq, input_lens, 50, db, bs, da)

      # Add predicted to list
      for i,sent in enumerate(predicted_sentences):
        all_predicted[batch_rows[i][-2]].append(sent) 

  json.dump(all_predicted, open('temp.json', 'w+'))
  time.sleep(2)

  if args.bs_predictor:
    bs_preds = np.concatenate([x[0] for x in bs_predictions])
    bs_true = np.concatenate([x[1] for x in bs_predictions])
    val_score = f1_score(bs_true, bs_preds, average="samples")
  elif args.dm_predictor:
    da_preds = np.concatenate([x[0] for x in bs_predictions])
    da_true = np.concatenate([x[1] for x in bs_predictions])
    val_score = f1_score(da_true, da_preds, average="samples")
  else:
    if args.domain:
      out = subprocess.check_output("python2.7 evaluate.py --pred temp.json --target temp_true.json".split())
    else:
      finished = False
      while not finished:
        try:
          out = subprocess.check_output("python2.7 evaluate.py --pred temp.json --target data/val_dials.json".split())
          finished = True
        except:
          print(":(")
          continue
    val_score = float(out.decode().split('|')[-1].strip())


  #val_score = evaluateModel(all_predicted, val_targets, mode='val')
  print("Epoch {0}: Validation Score {1:.10f}".format(epoch, val_score))
  print("-----------------------------------")
  if not (args.bs_predictor or args.dm_predictor):
    print(out.decode().split('|')[0] + '\n')

  if val_score > best_val_score:
    best_val_score = val_score
    best_val_epoch = epoch

if not args.test_only:
  print("Best validation score after epoch {0}".format(best_val_epoch))
  # Evaluate best val model on test data
  model.load(model_name+"_"+str(best_val_epoch))
else:
  model.load(model_name)
#lm.load('id_lm_10')
#model.lm = lm.decoder
#args.batch_size = 1
num_test_batches = math.ceil(len(test)/args.batch_size)
all_predicted = defaultdict(list)
bs_predictions = []
for batch in range(num_test_batches):
  indices = list(range(len(test)))
  if batch % 50 == 0:
    print("Batch {0}/{1}".format(batch, num_test_batches))
  # Prepare batch
  batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
  batch_rows = [test[i] for i in batch_indices]
  if args.bs_predictor:
    input_seq, input_lens, bs = model.prep_batch(batch_rows)
    predicted_bs = model.predict(input_seq, input_lens)
    bs_predictions.append((predicted_bs.data.cpu().numpy(), bs.data.cpu().numpy()))
  elif args.dm_predictor:
    bs, da, db = model.prep_batch(batch_rows)
    predicted_da = model.predict(bs, db)
    bs_predictions.append((predicted_da.data.cpu().numpy(), da.data.cpu().numpy()))
  elif args.nlg_predictor:
    target_seq, target_lens, db, da, bs = model.prep_batch(batch_rows)
    # Get predicted sentences for batch
    predicted_sentences = model.decode(50, db, da, bs)

    # Add predicted to list
    for i,sent in enumerate(predicted_sentences):
      all_predicted[batch_rows[i][-2]].append(sent) 
  else:
    input_seq, input_lens, target_seq, target_lens, db, bs, da = model.prep_batch(batch_rows)

    # Get predicted sentences for batch
    predicted_sentences = model.decode(input_seq, input_lens, 50, db, bs, da)
    #predicted_sentences = model.beam_decode(input_seq, input_lens, 50, db, bs, None)

    # Add predicted to list
    for i,sent in enumerate(predicted_sentences):
      all_predicted[batch_rows[i][-2]].append(sent) 

if args.bs_predictor:
  bs_preds = np.concatenate([x[0] for x in bs_predictions])
  bs_true = np.concatenate([x[1] for x in bs_predictions])
  test_score = f1_score(bs_true, bs_preds, average="samples")
  print("Test score:", test_score)
elif args.dm_predictor:
  da_preds = np.concatenate([x[0] for x in bs_predictions])
  da_true = np.concatenate([x[1] for x in bs_predictions])
  test_score = f1_score(da_true, da_preds, average="samples")
  print("Test score:", test_score)
else:
  json.dump(all_predicted, open('temp.json', 'w+'))
  time.sleep(2)
  out = subprocess.check_output("python2.7 evaluate.py --pred temp.json --target data/test_dials.json".split())
  print(out.decode().split('|')[0] + '\n')
  print("Test score:", float(out.decode().split('|')[-1].strip()))
