import argparse
import json
import math
import model
import random
import rl_eval
import sys
import torch

from collections import Counter

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

domains = [u'taxi',u'restaurant',  u'hospital', u'hotel',u'attraction', u'train', u'police']

parser = argparse.ArgumentParser(description='MultiWoz Training Script')

parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--use_attn', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--domain', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--train_lm', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--model_name', type=str, default='baseline')
parser.add_argument('--use_cuda', type=str2bool, default=True)
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


parser.add_argument('--load_lm', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--lm_name', type=str, default='baseline')
parser.add_argument('--s2s_name', type=str, default='baseline')

parser.add_argument('--load_nlu', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--load_dm', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--load_nlg', type=str2bool, const=True, nargs='?', default=False)
parser.add_argument('--tune_params', type=str2bool, const=True, nargs='?', default=True)
parser.add_argument('--multitasking', type=str2bool, const=True, nargs='?', default=False)


parser.add_argument('--data_size', type=float, default=-1.0)

args = parser.parse_args()

assert args.dm_predictor or args.bs_predictor or args.s2s_predictor or args.nlg_predictor or args.multitask_model or args.naive_fusion or args.structured_fusion, "Must turn on one training flag"


random.seed(args.seed)
torch.manual_seed(args.seed)

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

      # Add sys output
      input_so_far += target_seq

  return rows

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

def load_domain_data(filename, domains, exclude=False):

  data = json.load(open(filename))
  rows = []
  for file, dial in data.items():
    input_so_far = []

    for i in range(len(dial['sys'])):
      bs_doms = get_belief_state_domains(dial['bs'][i])
      if exclude is True:
        # Keep utterances which do not have any domains in 'domains'
        if len(list(set(bs_doms).intersection(domains))) > 0:
          continue
      else:
        # Keep utterances which have one of the domains in 'domains'
        if len(list(set(bs_doms).intersection(domains))) == 0:
          continue
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


      rows.append((input_seq, target_seq, db, bs, da))

      # Add sys output
      input_so_far += target_seq

  return rows

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

  dial_acts_dict = {e:i for i,e in enumerate([k for k,v in Counter(dial_acts).items() if v > 50  ])}
  print(dial_acts_dict, len(dial_acts_dict))
  return dial_acts_dict, data

# Load vocabulary
input_w2i = json.load(open('data/input_lang.word2index.json'))
output_w2i = json.load(open('data/output_lang.word2index.json'))
dial_act_dict, dial_acts_data = get_dial_acts('data/dialogue_act_feats.json')
# print(len(dial_act_dict), len(dial_acts_data))

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

nlg_decoder = model.Decoder(emb_size=args.emb_size,
                            hid_size=args.hid_size,
                            vocab_size=len(output_w2i),
                            use_attn=args.use_attn)


if args.bs_predictor:
  encoder = model.Encoder(vocab_size=len(input_w2i), 
                          emb_size=args.emb_size, 
                          hid_size=args.hid_size)
  model = model.NLU(encoder=encoder,
                    input_w2i=input_w2i,
                    args=args).cuda()
elif args.dm_predictor:
  pnn = model.PolicySmall(hidden_size=args.hid_size,
                          db_size=args.db_size,
                          bs_size=args.bs_size,
                          da_size=args.da_size)
  model = model.DM(pnn=pnn,
                   args=args).cuda()
elif args.nlg_predictor:
  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn,
                          concat_da=args.concat_da)
  model = model.NLG(decoder=decoder,
                    output_w2i=output_w2i,
                    args=args).cuda()
elif args.s2s_predictor:
  # Base components
  encoder = model.Encoder(vocab_size=len(input_w2i), 
                          emb_size=args.emb_size, 
                          hid_size=args.hid_size)

  #pnn = model.PNN(hidden_size=args.hid_size,
  #                db_size=args.db_size)

  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  #model = model.E2E(encoder=encoder,
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
                      args=args).cuda()

elif args.naive_fusion:
  # Base components
  encoder = model.Encoder(vocab_size=len(input_w2i),
                          emb_size=args.emb_size,
                          hid_size=args.hid_size)
  nlu = model.NLU(encoder=encoder,
                    input_w2i=input_w2i,
                    args=args)
  if args.load_nlu: 
    nlu.load('model/nlu_17')

  pnn = model.PolicySmall(hidden_size=args.hid_size,
                          db_size=args.db_size,
                          bs_size=args.bs_size,
                          da_size=args.da_size)
  dm = model.DM(pnn=pnn,
                args=args)
  if args.load_dm:
    dm.load('model/dm_4')

  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  nlg = model.NLG(decoder=decoder,
                    output_w2i=output_w2i,
                    args=args)
  if args.load_nlg:
    nlg.load('model/nlg_19')

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
  if args.load_nlu: 
    nlu.load('model/nlu_17')

  pnn = model.PolicyBig(hidden_size=args.hid_size,
                          db_size=args.db_size,
                          bs_size=args.bs_size,
                          da_size=args.da_size)
  dm = model.MultiTaskedDM(pnn=pnn,
                           args=args)
  if args.load_dm:
    dm.load('model/dm_4')


  decoder = model.Decoder(emb_size=args.emb_size,
                          hid_size=args.hid_size,
                          vocab_size=len(output_w2i),
                          use_attn=args.use_attn)
  nlg = model.NLG(decoder=decoder,
                    output_w2i=output_w2i,
                    args=args)
  if args.load_nlg:
    nlg.load('model/nlg_19')

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
  if args.load_nlu:
    nlu.load('model/nlu_17') 
  
  # DM
  dm_pnn = model.PolicySmall(hidden_size=args.hid_size,
                             db_size=args.db_size,
                             bs_size=args.bs_size,
                             da_size=args.da_size)
  dm = model.DM(pnn=dm_pnn,
                args=args).cuda()
  if args.load_dm:
    dm.load('model/dm_4')

  # NLG
  nlg_decoder = model.Decoder(emb_size=args.emb_size,
                              hid_size=args.hid_size,
                              vocab_size=len(output_w2i),
                              use_attn=args.use_attn)
  nlg= model.NLG(decoder=nlg_decoder,
                 output_w2i=output_w2i,
                 args=args) 
  if args.load_nlg:
    nlg.load('model/nlg_19')


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
  sl_model = model.StructuredFusion(nlu=nlu,
                                dm=dm,
                                nlg=nlg,
                                encoder=encoder,
                                pnn=pnn,
                                cf_dec=cf_dec,
                                decoder=decoder,
                                input_w2i=input_w2i,
                                output_w2i=output_w2i,
                                args=args).cuda()

  #sl_model.load('models/sfn_19')
  sl_model.load('models/multitask_fusion2_e2e_bs_4')

evaluator = rl_eval.MWEval()
agent = model.RLAgent(model=sl_model, evaluator=evaluator)

  
#if args.use_cuda is True:
#  model = model.cuda()

# Load data
train = load_data('data/train_dials.json', dial_acts_data, dial_act_dict)
valid = load_data('data/val_dials.json', dial_acts_data, dial_act_dict)
test = load_data('data/test_dials.json', dial_acts_data, dial_act_dict)

# Reduce data
if args.data_size > -1:
  random.shuffle(train)
  train = train[:int(len(train)*args.data_size)]

# Load domain data
if args.domain:
  test_domains = [u'restaurant']
  train = load_domain_data('data/train_dials.json', test_domains, exclude=True)
  valid = load_domain_data('data/val_dials.json', test_domains, exclude=True)
  test = load_domain_data('data/test_dials.json', test_domains, exclude=True)

print("Number of training instances:", len(train))
print("Number of validation instances:", len(valid))
print("Number of test instances:", len(test))


count = 0
for epoch in range(args.num_epochs):
  indices = list(range(len(train)))
  random.shuffle(indices)

  num_batches = math.ceil(len(train)/args.batch_size)
  cum_loss = 0
  for batch in range(num_batches):
    # Prepare batch
    batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
    batch_rows = [train[i] for i in batch_indices]

    #if batch < 50:
    #  agent.model.train(batch_rows)
    #else:
    cum_loss += agent.rl_train(batch_rows)

    # Log batch if needed
    if batch > 0 and batch % 50 == 0:
      print("Epoch {0}/{1} Batch {2}/{3} Avg Loss {4:.2f}".format(epoch+1, args.num_epochs, batch, num_batches, cum_loss/(batch+1)))

    if batch % 50 == 0:
      agent.model.save("{0}_{1}".format(args.model_name, count))
      count += 1
