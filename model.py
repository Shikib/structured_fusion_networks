import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
import numpy as np

import json
from collections import defaultdict

import random


#########################################################################################################
class FusionEncoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size, bs_size, embed=True):
    super(FusionEncoder, self).__init__() 
    self.embed = embed
    if self.embed:
      self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=3)
    self.encoder = nn.LSTM(emb_size+bs_size, hid_size)

  def pad(self, arr, pad=3):
    # Given an array of integer arrays, pad all arrays to the same length
    lengths = [len(e) for e in arr]
    max_len = max(lengths)
    return [e+[pad]*(max_len-len(e)) for e in arr], lengths

  def forward(self, seqs, lens, bs_preds):
    if type(seqs[0][0]) is int:
      seqs, lens = self.pad(seqs)
      seqs = torch.cuda.LongTensor(seqs).t()
      #if self.args.use_cuda is True: 
      #  seqs = torch.cuda.LongTensor(seqs).t()
      #else:
      #  seqs = torch.LongTensor(seqs).t()

    # Embed
    if self.embed:
      emb_seqs = self.embedding(seqs)
      emb_seqs = torch.cat((emb_seqs, bs_preds.repeat((emb_seqs.size(0), 1, 1))), dim=-1)
    else:
      emb_seqs = seqs

    # Sort by length
    sort_idx = sorted(range(len(lens)), key=lambda i: -lens[i])
    emb_seqs = emb_seqs[:,sort_idx]
    lens = [lens[i] for i in sort_idx]

    # Pack sequence
    packed = torch.nn.utils.rnn.pack_padded_sequence(emb_seqs, lens)

    # Forward pass through LSTM
    outputs, hidden = self.encoder(packed)

    # Unpack outputs
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    
    # Unsort
    unsort_idx = sorted(range(len(lens)), key=lambda i: sort_idx[i])
    outputs = outputs[:,unsort_idx]
    hidden = (hidden[0][:,unsort_idx], hidden[1][:,unsort_idx])

    return outputs, hidden

class FusionPolicy(nn.Module):
  def __init__(self, hidden_size, db_size, bs_size, da_size):
    super(FusionPolicy, self).__init__()
    self.proj_hid = nn.Linear(hidden_size, hidden_size)
    self.proj_db = nn.Linear(db_size, hidden_size)
    self.proj_bs = nn.Linear(bs_size, hidden_size)
    self.proj_da = nn.Linear(da_size, hidden_size)

  def forward(self, hidden, db, bs, da):
    output = self.proj_hid(hidden[0]) + self.proj_db(db) + self.proj_bs(bs) + self.proj_da(da)
    #output = self.proj_hid(hidden) + self.proj_db(db) + self.proj_da(da)
    return F.tanh(output)

class ColdFusionDecoder(nn.Module):
  def __init__(self, emb_size, hid_size, vocab_size, use_attn=True):
    super(ColdFusionDecoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    self.hid_size = hid_size
    if use_attn:
      self.decoder = nn.LSTM(vocab_size+emb_size+hid_size, hid_size)
      self.W_a = nn.Linear(hid_size * 2, hid_size)
      self.v = nn.Linear(hid_size, 1)
    else:
      self.decoder = nn.LSTM(vocab_size+emb_size, hid_size)

  def forward(self, hidden, last_word, encoder_outputs, lm_preds, ret_out=False):
    if not self.use_attn:
      embedded = self.embedding(last_word)
      rnn_input = torch.cat((lm_preds, embedded), dim=2)
      output, hidden = self.decoder(rnn_input, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output
    else:
      embedded = self.embedding(last_word)

      # Attn
      h = hidden[0].repeat(encoder_outputs.size(0), 1, 1)
      attn_energy = F.tanh(self.W_a(torch.cat((h, encoder_outputs), dim=2)))
      attn_logits = self.v(attn_energy).squeeze(-1) - 1e5 * (encoder_outputs.sum(dim=2) == 0).float()
      attn_weights = F.softmax(attn_logits, dim=0).permute(1,0).unsqueeze(1)
      context_vec = attn_weights.bmm(encoder_outputs.permute(1,0,2)).permute(1,0,2)

      # Concat with embeddings
      rnn_input = torch.cat((context_vec, embedded, lm_preds.unsqueeze(0)), dim=2)

      # Forward
      output, hidden = self.decoder(rnn_input, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output

class PolicySmall(nn.Module):
  def __init__(self, hidden_size, db_size, bs_size, da_size):
    super(PolicySmall, self).__init__()
    self.proj_db = nn.Linear(db_size, hidden_size)
    self.proj_bs = nn.Linear(bs_size, hidden_size)

  def forward(self, db, bs):
    output = self.proj_db(db) + self.proj_bs(bs)
    return F.relu(output)

class PolicyBig(nn.Module):
  def __init__(self, hidden_size, db_size, bs_size, da_size):
    super(PolicyBig, self).__init__()
    self.proj_db = nn.Linear(db_size, hidden_size)
    self.proj_hid = nn.Linear(hidden_size, hidden_size)

  def forward(self, hid, db):
    output = self.proj_hid(hid) + self.proj_db(db)
    return F.relu(output)
#########################################################################################################

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size, embed=True):
    super(Encoder, self).__init__() 
    self.embed = embed
    if self.embed:
      self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=3)
    self.encoder = nn.LSTM(emb_size, hid_size)

  def pad(self, arr, pad=3):
    # Given an array of integer arrays, pad all arrays to the same length
    lengths = [len(e) for e in arr]
    max_len = max(lengths)
    return [e+[pad]*(max_len-len(e)) for e in arr], lengths

  def forward(self, seqs, lens):
    if type(seqs[0][0]) is int:
      seqs, lens = self.pad(seqs)
      seqs = torch.cuda.LongTensor(seqs).t()
      #if self.args.use_cuda is True: 
      #  seqs = torch.cuda.LongTensor(seqs).t()
      #else:
      #  seqs = torch.LongTensor(seqs).t()

    # Embed
    if self.embed:
      emb_seqs = self.embedding(seqs)
    else:
      emb_seqs = seqs

    # Sort by length
    sort_idx = sorted(range(len(lens)), key=lambda i: -lens[i])
    emb_seqs = emb_seqs[:,sort_idx]
    lens = [lens[i] for i in sort_idx]

    # Pack sequence
    packed = torch.nn.utils.rnn.pack_padded_sequence(emb_seqs, lens)

    # Forward pass through LSTM
    outputs, hidden = self.encoder(packed)

    # Unpack outputs
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    
    # Unsort
    unsort_idx = sorted(range(len(lens)), key=lambda i: sort_idx[i])
    outputs = outputs[:,unsort_idx]
    hidden = (hidden[0][:,unsort_idx], hidden[1][:,unsort_idx])

    return outputs, hidden

class HierarchicalEncoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size):
    super(HierarchicalEncoder, self).__init__() 
    self.utt_encoder = Encoder(vocab_size, emb_size, hid_size, embed=True)
    self.ctx_encoder = Encoder(vocab_size, hid_size, hid_size, embed=False)

  def forward(self, seqs, lens):
    # First split into utterances
    utts = []
    conv_inds = []
    for i,seq in enumerate(seqs):
      cur_seq = seq
      while len(cur_seq) > 0:
        # Find and add full utt
        next_utt_ind = cur_seq.index(1)+1
        utts.append(cur_seq[:next_utt_ind])
        cur_seq = cur_seq[next_utt_ind:]
        conv_inds.append(i)

    # Encode all of the utterances
    _, encoder_hiddens = self.utt_encoder(utts, None)

    # Re-construct conversations
    ctx_hiddens = [[] for _ in range(len(lens))]
    for i,ind in enumerate(conv_inds):
      ctx_hiddens[ind].append(encoder_hiddens[0][0,i])

    # Pad hidden states and create a tensor
    ctx_lens = [len(ctx) for ctx in ctx_hiddens]
    max_ctx_len = max(ctx_lens)
    hid_size = ctx_hiddens[0][0].size()
    ctx_hiddens = [ctx+[torch.zeros(hid_size).cuda()]*(max_ctx_len-len(ctx))
                   for ctx in ctx_hiddens]
    ctx_tensor = torch.stack([torch.stack(ctx) for ctx in ctx_hiddens]).permute(1,0,2)

    return self.ctx_encoder(ctx_tensor, ctx_lens)

class Policy(nn.Module):
  def __init__(self, hidden_size, db_size, bs_size, da_size):
    super(Policy, self).__init__()
    self.proj_hid = nn.Linear(hidden_size, hidden_size)
    self.proj_db = nn.Linear(db_size, hidden_size)
    self.proj_bs = nn.Linear(bs_size, hidden_size)
    #self.proj_da = nn.Linear(da_size, hidden_size)

  def forward(self, hidden, db, bs, da):
    output = self.proj_hid(hidden[0]) + self.proj_db(db) + self.proj_bs(bs) #+ self.proj_da(da)
    return (F.tanh(output), hidden[1])

class PNN(nn.Module):
  def __init__(self, hidden_size, db_size):
    super(PNN, self).__init__()
    self.proj_hid = nn.Linear(hidden_size, hidden_size)
    self.proj_db = nn.Linear(db_size, hidden_size)
    self.proj_bs = nn.Linear(94, hidden_size)
    
    self.layers = nn.ModuleList()
    for i in range(3):
      self.layers.append(nn.Linear(hidden_size, hidden_size))

    self.final = nn.Linear(hidden_size, hidden_size)

  def forward(self, hidden, db, bs):
    output = F.relu(self.proj_hid(hidden) + self.proj_db(db) + self.proj_bs(bs))
    for l in self.layers:
      output = F.relu(l(output))
    return F.relu(self.final(output))

class Decoder(nn.Module):
  def __init__(self, emb_size, hid_size, vocab_size, use_attn=True, concat_da=False):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.out = nn.Linear(hid_size, vocab_size)
    self.use_attn = use_attn
    self.hid_size = hid_size
    self.concat_da = concat_da
    if use_attn:
      if concat_da:
        self.decoder = nn.LSTM(emb_size+2*hid_size, hid_size)
      else:
        self.decoder = nn.LSTM(emb_size+hid_size, hid_size)
      self.W_a = nn.Linear(hid_size * 2, hid_size)
      self.v = nn.Linear(hid_size, 1)
    else:
      self.decoder = nn.LSTM(emb_size, hid_size)

  def forward(self, hidden, last_word, encoder_outputs, da=None, ret_out=False):
    if not self.use_attn:
      embedded = self.embedding(last_word)
      #output, hidden = self.decoder(torch.cat((embedded, concat), dim=-1), hidden)
      output, hidden = self.decoder(embedded, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output
    else:
      embedded = self.embedding(last_word)

      if self.concat_da:
        embedded = torch.cat((embedded, da), dim=2)
 
      # Attn
      h = hidden[0].repeat(encoder_outputs.size(0), 1, 1)
      attn_energy = F.tanh(self.W_a(torch.cat((h, encoder_outputs), dim=2)))
      attn_logits = self.v(attn_energy).squeeze(-1) - 1e5 * (encoder_outputs.sum(dim=2) == 0).float()
      attn_weights = F.softmax(attn_logits, dim=0).permute(1,0).unsqueeze(1)
      context_vec = attn_weights.bmm(encoder_outputs.permute(1,0,2)).permute(1,0,2)

      # Concat with embeddings
      rnn_input = torch.cat((context_vec, embedded), dim=2)

      # Forward
      output, hidden = self.decoder(rnn_input, hidden)
      if not ret_out:
        return F.log_softmax(self.out(output), dim=2), hidden
      else:
        return F.log_softmax(self.out(output), dim=2), hidden, output


class ColdFusionLayer(nn.Module):
  def __init__(self, hid_size, vocab_size):
    super(ColdFusionLayer, self).__init__()
    self.lm_proj = nn.Linear(vocab_size, hid_size)
    self.mask_proj = nn.Linear(2*hid_size, hid_size)
    self.out = nn.Linear(2*hid_size, vocab_size)

  def forward(self, dec_hid, lm_pred, pred=True):
    lm_hid = self.lm_proj(lm_pred)
    mask = F.sigmoid(self.mask_proj(torch.cat((dec_hid, lm_hid), dim=2)))

    if pred:
      return F.log_softmax(self.out(torch.cat((dec_hid, lm_hid*mask), dim=2)), dim=2)
    else:
      return self.out(torch.cat((dec_hid, lm_hid*mask), dim=2))

class Model(nn.Module):
  def __init__(self, encoder, policy, decoder, input_w2i, output_w2i, args):
    super(Model, self).__init__()

    self.args = args

    # Model
    self.encoder = encoder
    self.policy = policy
    self.decoder = decoder

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=True):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    # if self.args.use_cuda is True:
    #   input_seq = torch.cuda.LongTensor(input_seq).t()
    # else:
    #   input_seq = torch.LongTensor(input_seq).t()
    input_seq = inputs
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs, da

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs, da):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Policy network
    encoder_hidden = self.policy(encoder_hidden, db, bs, da)
    decoder_hidden = encoder_hidden

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, encoder_hidden[0])

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs, da):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs, da)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs, da):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs, da)
      encoder_hidden = self.policy(encoder_hidden, db, bs, da)

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, encoder_hidden[0])

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), encoder_outputs)
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], encoder_outputs)

          # Get top candidates and add new hypotheses
          topv, topi = decoder_output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())

class LanguageModel(nn.Module):
  def __init__(self, decoder, input_w2i, output_w2i, args):
    super(LanguageModel, self).__init__()

    self.args = args
    assert not decoder.use_attn

    # Model
    self.decoder = decoder

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=True):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    # if self.args.use_cuda is True:
    #   input_seq = torch.cuda.LongTensor(input_seq).t()
    # else:
    #   input_seq = torch.LongTensor(input_seq).t()
    input_seq = inputs
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    decoder_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                      torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None)

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      decoder_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                        torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None)

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      decoder_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                        torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), None)
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], None)

          # Get top candidates and add new hypotheses
          topv, topi = decoder_output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.decoder, name+'.dec')

  def load(self, name):
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())

class ShallowFusionModel(nn.Module):
  def __init__(self, seq2seq, lm, args):
    super(ShallowFusionModel, self).__init__()

    self.args = args

    # Model
    self.encoder = seq2seq.encoder
    self.policy = seq2seq.policy
    self.decoder = seq2seq.decoder
    self.lm = lm.decoder

    # Vocab
    self.input_i2w = seq2seq.input_i2w
    self.input_w2i = seq2seq.input_w2i
    self.output_i2w = seq2seq.output_i2w
    self.output_w2i = seq2seq.output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=False):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    # if self.args.use_cuda is True:
    #   input_seq = torch.cuda.LongTensor(input_seq).t()
    # else:
    #   input_seq = torch.LongTensor(input_seq).t()
    input_seq = inputs
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Policy network
    decoder_hidden = self.policy(encoder_hidden, db, bs)

    # LM hidden
    lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                 torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)
      lm_output, lm_hidden = self.lm(lm_hidden, last_word, None)

      # Save output
      probas[t] = decoder_output + lm_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs)
        lm_output, lm_hidden = self.lm(lm_hidden, last_word, None)

        # Get top candidates
        topv, topi = (decoder_output + lm_output).data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, lm_hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())


      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      beam = [(_to_cpu(decoder_hidden), _to_cpu(lm_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[2]), encoder_outputs)
            lm_output, lm_hidden = self.lm(_to_cuda(hyp[1]), _to_cuda(hyp[2]), None)
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[2], encoder_outputs)
            lm_output, lm_hidden = self.lm(hyp[1], hyp[2], None)

          # Get top candidates and add new hypotheses
          topv, topi = (decoder_output+lm_output).data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(lm_hidden),
                       _to_cpu(last_word), 
                       hyp[3] + topv[i], 
                       hyp[4] + 1, 
                       hyp[5] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      finished = [hyp for hyp in beam if hyp[-1].endswith('_EOS')]
      if len(finished) == 0:
        finished = beam
      return [max(finished, key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')
    torch.save(self.lm, name+'.lmdec')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
    self.lm.load_state_dict(torch.load(name+'.lmdec').state_dict())

class DeepFusionModel(nn.Module):
  def __init__(self, seq2seq, lm, args):
    super(DeepFusionModel, self).__init__()

    self.args = args

    # Model
    self.encoder = seq2seq.encoder
    self.policy = seq2seq.policy
    self.decoder = seq2seq.decoder
    self.lm = lm.decoder
    self.out = nn.Linear(self.decoder.hid_size + self.lm.hid_size, len(seq2seq.output_w2i))

    # Vocab
    self.input_i2w = seq2seq.input_i2w
    self.input_w2i = seq2seq.input_w2i
    self.output_i2w = seq2seq.output_i2w
    self.output_w2i = seq2seq.output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=False):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    # if self.args.use_cuda is True:
    #   input_seq = torch.cuda.LongTensor(input_seq).t()
    # else:
    #   input_seq = torch.LongTensor(input_seq).t()
    input_seq = inputs
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Policy network
    decoder_hidden = self.policy(encoder_hidden, db, bs)

    # LM hidden
    lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                 torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
      _, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

      # Save output
      probas[t] = F.log_softmax(self.out(F.dropout(torch.cat((decoder_output.detach(), lm_output.detach()), dim=2), p=0.5)), dim=2)

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
        _, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

        # Save output
        output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)

        # Get top candidates
        topv, topi = output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, lm_hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())


      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      beam = [(_to_cpu(decoder_hidden), _to_cpu(lm_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            _, decoder_hidden, decoder_output = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[2]), encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(_to_cuda(hyp[1]), _to_cuda(hyp[2]), None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)
          else:
            _, decoder_hidden, decoder_output = self.decoder(hyp[0], hyp[2], encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(hyp[1], hyp[2], None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)

          # Get top candidates and add new hypotheses
          topv, topi = output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(lm_hidden),
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')
    torch.save(self.lm, name+'.lmdec')
    torch.save(self.out, name+'.out')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
    self.lm.load_state_dict(torch.load(name+'.lmdec').state_dict())
    self.out.load_state_dict(torch.load(name+'.out').state_dict())

class ColdFusionModel(nn.Module):
  def __init__(self, seq2seq, lm, cf, args):
    super(ColdFusionModel, self).__init__()

    self.args = args

    # Model
    self.encoder = seq2seq.encoder
    self.policy = seq2seq.policy
    self.decoder = seq2seq.decoder
    self.lm = lm.decoder
    self.cf = cf

    # Vocab
    self.input_i2w = seq2seq.input_i2w
    self.input_w2i = seq2seq.input_w2i
    self.output_i2w = seq2seq.output_i2w
    self.output_w2i = seq2seq.output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, hierarchical=False):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
    # if self.args.use_cuda is True:
    #   input_seq = torch.cuda.LongTensor(input_seq).t()
    # else:
    #   input_seq = torch.LongTensor(input_seq).t()
    input_seq = inputs
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # Policy network
    decoder_hidden = self.policy(encoder_hidden, db, bs)

    # LM hidden
    lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                 torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
      lm_pred, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

      # Save output
      probas[t] = self.cf(decoder_output, lm_pred.detach())

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        _, decoder_hidden, decoder_output = self.decoder(decoder_hidden, last_word, encoder_outputs, ret_out=True)
        lm_pred, lm_hidden, lm_output = self.lm(lm_hidden, last_word, None, ret_out=True)

        # Save output
        output = self.cf(decoder_output, lm_pred.detach())

        # Get top candidates
        topv, topi = output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, lm_hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = self.policy(encoder_hidden, db, bs)

      # LM hidden
      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())


      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      beam = [(_to_cpu(decoder_hidden), _to_cpu(lm_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            _, decoder_hidden, decoder_output = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[2]), encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(_to_cuda(hyp[1]), _to_cuda(hyp[2]), None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)
          else:
            _, decoder_hidden, decoder_output = self.decoder(hyp[0], hyp[2], encoder_outputs, ret_out=True)
            _, lm_hidden, lm_output = self.lm(hyp[1], hyp[2], None, ret_out=True)
            output = F.log_softmax(self.out(torch.cat((decoder_output, lm_output), dim=2)), dim=2)

          # Get top candidates and add new hypotheses
          topv, topi = output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(lm_hidden),
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]

      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self.encoder, name+'.enc')
    torch.save(self.policy, name+'.pol')
    torch.save(self.decoder, name+'.dec')
    torch.save(self.lm, name+'.lmdec')
    torch.save(self.cf, name+'.cf')

  def load(self, name):
    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
    self.lm.load_state_dict(torch.load(name+'.lmdec').state_dict())
    self.cf.load_state_dict(torch.load(name+'.cf').state_dict())




#class ColdFusionModel(nn.Module):
#  def __init__(self, encoder, policy, decoder, lm, input_w2i, output_w2i, args):
#    super(ColdFusionModel, self).__init__()
#
#    self.args = args
#
#    # Model
#    self.encoder = encoder
#    self.policy = policy
#    self.decoder = decoder
#    self.lm = lm
#
#    # Vocab
#    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
#    self.input_w2i = input_w2i
#    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
#    self.output_w2i = output_w2i
#
#    # Training
#    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
#    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
#
#  def prep_batch(self, rows, hierarchical=True):
#    def _pad(arr, pad=3):
#      # Given an array of integer arrays, pad all arrays to the same length
#      lengths = [len(e) for e in arr]
#      max_len = max(lengths)
#      return [e+[pad]*(max_len-len(e)) for e in arr], lengths
#
#    inputs = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
#    # input_seq, input_lens = _pad(inputs, pad=self.input_w2i['_PAD'])
#    # if self.args.use_cuda is True:
#    #   input_seq = torch.cuda.LongTensor(input_seq).t()
#    # else:
#    #   input_seq = torch.LongTensor(input_seq).t()
#    input_seq = inputs
#    input_lens = [len(inp) for inp in input_seq]
#
#    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
#    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
#    if self.args.use_cuda is True:
#      target_seq = torch.cuda.LongTensor(target_seq).t()
#    else:
#      target_seq = torch.LongTensor(target_seq).t()
#
#    if self.args.use_cuda is True:
#      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
#      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
#    else:
#      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
#      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
#
#    return input_seq, input_lens, target_seq, target_lens, db, bs
#
#  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs):
#    # Pass through LM
#    lm_probas = self.lm(input_seq, input_lens, target_seq, target_lens, db, bs).detach()
#
#    # Encoder
#    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
#
#    # Policy network
#    decoder_hidden = self.policy(encoder_hidden, db, bs)
#
#    # Decoder
#    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
#    if self.args.use_cuda is True:
#      probas = probas.cuda()
#    last_word = target_seq[0].unsqueeze(0)
#    for t in range(1,target_seq.size(0)):
#      # Pass through decoder
#      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, lm_probas[t])
#
#      # Save output
#      probas[t] = decoder_output
#
#      # Set new last word
#      last_word = target_seq[t].unsqueeze(0)
#
#    return lm_probas
#
#  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs):
#    self.optim.zero_grad()
#
#    # Forward
#    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)
#
#    # Loss
#    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
#
#    # Backwards
#    loss.backward()
#    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
#    self.optim.step()
#
#    return loss.item()
#
#  def decode(self, input_seq, input_lens, max_len, db, bs):
#
#    batch_size = len(input_seq)
#    predictions = torch.zeros((batch_size, max_len))
#
#    with torch.no_grad():
#      # Encoder
#      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
#
#      # Policy network
#      decoder_hidden = self.policy(encoder_hidden, db, bs)
#      lm_hidden = (torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda(),
#                   torch.zeros((1, len(input_seq), self.decoder.hid_size)).cuda())
#
#      # Decoder
#      if self.args.use_cuda is True:
#        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      else:
#        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      for t in range(max_len):
#        lm_probas, lm_hidden = self.lm.decoder(lm_hidden, last_word, None)
#
#        # Pass through decoder
#        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, lm_probas.squeeze(0))
#
#        # Get top candidates
#        topv, topi = decoder_output.data.topk(1)
#        topi = topi.view(-1)
#
#        predictions[:, t] = topi
#
#        # Set new last word
#        last_word = topi.detach().view(1, -1)
#
#    predicted_sentences = []
#    for sentence in predictions:
#      sent = []
#      for ind in sentence:
#        word = self.output_i2w[ind.long().item()]
#        if word == '_EOS':
#          break
#        sent.append(word)
#      predicted_sentences.append(' '.join(sent))
#
#    return predicted_sentences
#
#  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
#    assert False, "not implemented yet"
#    def _to_cpu(x):
#      if type(x) in [tuple, list]:
#        return [e.cpu() for e in x]
#      else:
#        return x.cpu()
#
#    def _to_cuda(x):
#      if type(x) in [tuple, list]:
#        return [e.cuda() for e in x]
#      else:
#        return x.cuda()
#
#    def _score(hyp):
#      return hyp[2]/float(hyp[3] + 1e-6) 
#
#    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
#    with torch.no_grad():
#      # Batch size must be 1
#      assert len(input_seq) == 1
#
#      # Encoder
#      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
#
#      # Policy network
#      decoder_hidden = self.policy(encoder_hidden, db, bs)
#
#      # Decoder
#      if self.args.use_cuda is True:
#        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      else:
#        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
#      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
#      for _ in range(max_len):
#        new_beam = []
#        for hyp in beam:
#          # Continue _EOS
#          if hyp[-1].endswith('_EOS'):
#            new_beam.append(hyp)
#            continue
#          
#          # Propagate through decoder
#          if self.args.use_cuda is True:
#            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), encoder_outputs)
#          else:
#            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], encoder_outputs)
#
#          # Get top candidates and add new hypotheses
#          topv, topi = decoder_output.data.topk(beam_width)
#          topv = topv.squeeze()
#          topi = topi.squeeze()
#          for i in range(beam_width):
#            last_word = topi[i].detach().view(1, -1)
#            new_hyp = (_to_cpu(decoder_hidden), 
#                       _to_cpu(last_word), 
#                       hyp[2] + topv[i], 
#                       hyp[3] + 1, 
#                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])
#
#            new_beam.append(new_hyp)
#
#        # Translate new beam into beam
#        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]
#
#      return [max([hyp for hyp in beam if hyp[-1].endswith('_EOS')], key=_score)[-1].replace("_EOS", "").strip()]
#          
#  def save(self, name):
#    torch.save(self.encoder, name+'.enc')
#    torch.save(self.policy, name+'.pol')
#    torch.save(self.decoder, name+'.dec')
#    torch.save(self.lm, name+'.lm')
#
#  def load(self, name):
#    self.encoder.load_state_dict(torch.load(name+'.enc').state_dict())
#    self.policy.load_state_dict(torch.load(name+'.pol').state_dict())
#    self.decoder.load_state_dict(torch.load(name+'.dec').state_dict())
#    self.lm.load_state_dict(torch.load(name+'.lm').state_dict())

class NLU(nn.Module):
  def __init__(self, encoder, input_w2i, args):
    super(NLU, self).__init__() 

    # Model
    self.encoder = encoder
    self.linear = nn.Linear(args.hid_size, args.bs_size)

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i

    # Training
    self.criterion = nn.BCEWithLogitsLoss(reduce=True, pos_weight=torch.Tensor([10.0]))
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    input_seq = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    input_lens = [len(inp) for inp in input_seq]

    if self.args.use_cuda is True:
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, bs

  def forward(self, input_seq, input_lens):
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)
    bs_out_logits = self.linear(encoder_hidden[0])

    return bs_out_logits.squeeze(0)

  def train(self, input_seq, input_lens, bs, no_prop=False):
    self.optim.zero_grad()

    # Forward
    bs_out_logits = self.forward(input_seq, input_lens)

    # Loss
    loss = self.criterion(bs_out_logits, bs)

    if no_prop:
      return loss

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def predict(self, input_seq, input_lens):
    out_logits = self.forward(input_seq, input_lens)
    out_probs = F.sigmoid(out_logits)
    return out_probs >= 0.5

  def save(self, name):
    torch.save(self, name+'.nlu')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.nlu').state_dict())

class DM(nn.Module):
  def __init__(self, pnn, args):
    super(DM, self).__init__() 

    # Model
    self.bs_in = nn.Linear(args.bs_size, args.hid_size)
    self.pnn = pnn
    self.da_out = nn.Linear(args.hid_size, args.da_size)

    # Training
    self.criterion = nn.BCEWithLogitsLoss(reduce=True, pos_weight=torch.Tensor([10.0]))
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows):
    if self.args.use_cuda is True:
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
    else:
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.FloatTensor([[int(e) for e in row[4]] for row in rows])
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])

    return bs, da, db

  def forward(self, bs, db):
    da_hid = self.pnn(db, bs)
    da_pred = self.da_out(da_hid)
    return da_pred

  def train(self, bs, da, db, no_prop=False):
    self.optim.zero_grad()

    # Forward
    logits = self.forward(bs, db)

    # Loss
    loss = self.criterion(logits, da)
    if no_prop:
      return loss

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def predict(self, bs, db):
    out_logits = self.forward(bs, db)
    out_probs = F.sigmoid(out_logits)
    return out_probs >= 0.5

  def save(self, name):
    torch.save(self, name+'.dm')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.dm').state_dict())

class MultiTaskedDM(nn.Module):
  def __init__(self, pnn, args):
    super(MultiTaskedDM, self).__init__() 

    # Model
    self.bs_in = nn.Linear(args.bs_size, args.hid_size)
    self.pnn = pnn
    self.da_out = nn.Linear(args.hid_size, args.da_size)

    # Training
    self.criterion = nn.BCEWithLogitsLoss(reduce=True, pos_weight=torch.Tensor([10.0]))
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows):
    if self.args.use_cuda is True:
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
    else:
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.FloatTensor([[int(e) for e in row[4]] for row in rows])
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])

    return bs, da, db

  def forward(self, bs, db):
    hid = F.relu(self.bs_in(bs))
    da_hid = self.pnn(hid, db)
    da_pred = self.da_out(da_hid)
    return da_pred

  def train(self, bs, da, db, no_prop=False):
    self.optim.zero_grad()

    # Forward
    logits = self.forward(bs, db)

    # Loss
    loss = self.criterion(logits, da)
    if no_prop:
      return loss

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def predict(self, bs, db):
    out_logits = self.forward(bs, db)
    out_probs = F.sigmoid(out_logits)
    return out_probs >= 0.5

  def save(self, name):
    torch.save(self, name+'.dm')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.dm').state_dict())

class NLG(nn.Module):
  def __init__(self, decoder, output_w2i, args):
    super(NLG, self).__init__()

    # Model
    self.da_in = nn.Linear(args.da_size, args.hid_size)
    self.db_in = nn.Linear(args.db_size, args.hid_size)
    self.bs_in = nn.Linear(args.bs_size, args.hid_size)
    self.da_inc = nn.Linear(args.da_size, args.hid_size)
    self.db_inc = nn.Linear(args.db_size, args.hid_size)
    self.bs_inc = nn.Linear(args.bs_size, args.hid_size)
    self.decoder = decoder

    # Vocab
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=output_w2i['_PAD'], size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])

    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.FloatTensor([[int(e) for e in row[4]] for row in rows])

    return target_seq, target_lens, db, da, bs

  def forward(self, target_seq, target_lens, db, da, bs):
    da_proj = self.da_in(da).unsqueeze(0)
    db_proj = self.db_in(db).unsqueeze(0)
    bs_proj = self.bs_in(bs).unsqueeze(0)
    da_projc = self.da_inc(da).unsqueeze(0)
    db_projc = self.db_inc(db).unsqueeze(0)
    bs_projc = self.bs_inc(bs).unsqueeze(0)

    # Policy network
    decoder_hidden = (F.relu(da_proj + db_proj + bs_proj), F.relu(da_projc+db_projc + bs_projc))
    encoder_hidden = decoder_hidden

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None, encoder_hidden[0])

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def train(self, target_seq, target_lens, db, da, bs, no_prop=False):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(target_seq, target_lens, db, da, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
    if no_prop:
      return loss

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, max_len, db, da, bs):
    batch_size = len(db)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      da_proj = self.da_in(da).unsqueeze(0)
      db_proj = self.db_in(db).unsqueeze(0)
      bs_proj = self.bs_in(bs).unsqueeze(0)
      da_projc = self.da_inc(da).unsqueeze(0)
      db_projc = self.db_inc(db).unsqueeze(0)
      bs_projc = self.bs_inc(bs).unsqueeze(0)

      # Policy network
      decoder_hidden = (F.relu(da_proj + db_proj + bs_proj), F.relu(da_projc+db_projc + bs_projc))
      encoder_hidden = decoder_hidden

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(batch_size)]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(batch_size)]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, None, encoder_hidden[0])

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def save(self, name):
    torch.save(self, name+'.nlg')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.nlg').state_dict())

class E2E(nn.Module):
  def __init__(self, encoder, pnn, decoder, input_w2i, output_w2i, args):
    super(E2E, self).__init__()

    # Model
    self.encoder = encoder
    self.pnn = pnn
    self.decoder = decoder

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows, hierarchical=True):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    input_seq = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs, auxillary=False):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    pre_hidden = encoder_hidden

    # Policy network
    decoder_hidden = (self.pnn(encoder_hidden[0], db), encoder_hidden[1].cuda())
    encoder_hidden = decoder_hidden

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w))
    if self.args.use_cuda is True:
      probas = probas.cuda()
    last_word = target_seq[0].unsqueeze(0)
    for t in range(1,target_seq.size(0)):
      # Pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, encoder_hidden[0]) 

      # Save output
      probas[t] = decoder_output

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    if auxillary:
      return probas, pre_hidden[0], encoder_hidden[0]
    else:
      return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs, no_prop=False):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
    if no_prop:
      return loss

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = (self.pnn(encoder_hidden[0], db), encoder_hidden[1].cuda())
      encoder_hidden = decoder_hidden
      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      for t in range(max_len):
        # Pass through decoder
        decoder_output, decoder_hidden = self.decoder(decoder_hidden, last_word, encoder_outputs, encoder_hidden[0]) 

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = (self.pnn(encoder_hidden[0], db), torch.zeros(encoder_hidden[0].size()).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), encoder_outputs, self.pnn(encoder_hidden[0], db))
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], encoder_outputs)

          # Get top candidates and add new hypotheses
          topv, topi = decoder_output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]
      eos_end = [hyp for hyp in beam if hyp[-1].endswith('_EOS')]
      if len(eos_end) == 0:
        eos_end = beam
      return [max(eos_end, key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self, name+'.e2e')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.e2e').state_dict())

class NaiveFusion(nn.Module):
  def __init__(self, nlu, dm, nlg, input_w2i, output_w2i, args):
    super(NaiveFusion, self).__init__()

    # Model
    self.nlu = nlu
    self.dm = dm
    self.nlg = nlg

    # Vocab
    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    # Training
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def prep_batch(self, rows, hierarchical=True):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    input_seq = [[self.input_w2i.get(w, self.input_w2i['_UNK']) for w in row[0]] for row in rows]
    input_lens = [len(inp) for inp in input_seq]

    targets = [[self.output_w2i.get(w, self.output_w2i['_UNK']) for w in row[1]] for row in rows]
    target_seq, target_lens = _pad(targets, pad=self.output_w2i['_PAD'])
    if self.args.use_cuda is True:
      target_seq = torch.cuda.LongTensor(target_seq).t()
    else:
      target_seq = torch.LongTensor(target_seq).t()

    if self.args.use_cuda is True:
      db = torch.cuda.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.cuda.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])
    else:
      db = torch.FloatTensor([[int(e) for e in row[2]] for row in rows])
      bs = torch.FloatTensor([[int(e) for e in row[3]] for row in rows])
      da = torch.cuda.FloatTensor([[int(e) for e in row[4]] for row in rows])

    return input_seq, input_lens, target_seq, target_lens, db, bs, da

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs, da):
    # NLU
    bs = F.sigmoid(self.nlu(input_seq, input_lens))

    # DM
    da = F.sigmoid(self.dm(bs, db))

    # NLG
    probas = self.nlg(target_seq, target_lens, db, da, bs)

    return probas

  def train(self, input_seq, input_lens, target_seq, target_lens, db, bs, da, no_prop=False):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs, da)

    # Loss
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
    if no_prop:
      return loss

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def decode(self, input_seq, input_lens, max_len, db, bs, da):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    with torch.no_grad():
      # NLU
      bs = (F.sigmoid(self.nlu(input_seq, input_lens)) >= 0.5).float()

      # DM
      da = (F.sigmoid(self.dm(bs, db)) >= 0.5).float()

      # NLG
      return self.nlg.decode(max_len, db, da, bs)

  def beam_decode(self, input_seq, input_lens, max_len, db, bs, beam_width=10):
    def _to_cpu(x):
      if type(x) in [tuple, list]:
        return [e.cpu() for e in x]
      else:
        return x.cpu()

    def _to_cuda(x):
      if type(x) in [tuple, list]:
        return [e.cuda() for e in x]
      else:
        return x.cuda()

    def _score(hyp):
      return hyp[2]/float(hyp[3] + 1e-6) 

    # Beam is (hid_cpu, input_word, log_p, length, seq_so_far)
    with torch.no_grad():
      # Batch size must be 1
      assert len(input_seq) == 1

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # Policy network
      decoder_hidden = (self.pnn(encoder_hidden[0], db), torch.zeros(encoder_hidden[0].size()).cuda())

      # Decoder
      if self.args.use_cuda is True:
        last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      else:
        last_word = torch.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])
      beam = [(_to_cpu(decoder_hidden), _to_cpu(last_word), 0, 0, "")]
      for _ in range(max_len):
        new_beam = []
        for hyp in beam:
          # Continue _EOS
          if hyp[-1].endswith('_EOS'):
            new_beam.append(hyp)
            continue
          
          # Propagate through decoder
          if self.args.use_cuda is True:
            decoder_output, decoder_hidden = self.decoder(_to_cuda(hyp[0]), _to_cuda(hyp[1]), encoder_outputs, self.pnn(encoder_hidden[0], db))
          else:
            decoder_output, decoder_hidden = self.decoder(hyp[0], hyp[1], encoder_outputs)

          # Get top candidates and add new hypotheses
          topv, topi = decoder_output.data.topk(beam_width)
          topv = topv.squeeze()
          topi = topi.squeeze()
          for i in range(beam_width):
            last_word = topi[i].detach().view(1, -1)
            new_hyp = (_to_cpu(decoder_hidden), 
                       _to_cpu(last_word), 
                       hyp[2] + topv[i], 
                       hyp[3] + 1, 
                       hyp[4] + " " + self.output_i2w[topi[i].long().item()])

            new_beam.append(new_hyp)

        # Translate new beam into beam
        beam = sorted(new_beam, key=_score, reverse=True)[:beam_width]
      eos_end = [hyp for hyp in beam if hyp[-1].endswith('_EOS')]
      if len(eos_end) == 0:
        eos_end = beam
      return [max(eos_end, key=_score)[-1].replace("_EOS", "").strip()]
          
  def save(self, name):
    torch.save(self, name+'.e2e')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.e2e').state_dict())

class MultiTask(nn.Module):
  def __init__(self, nlu, dm, nlg, e2e, args):
    super(MultiTask, self).__init__()

    # Model
    self.nlu = nlu
    self.dm = dm
    self.nlg = nlg
    self.e2e = e2e

    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.args = args

  def train(self, batch_rows):
    self.optim.zero_grad()

    # Train NLU
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    nlu_loss = self.nlu.train(input_seq, input_lens, bs, no_prop=True)

    # Train DM
    bs, da, db = self.dm.prep_batch(batch_rows)
    dm_loss = self.dm.train(bs, da, db, no_prop=True)

    # Train NLG
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)
    nlg_loss = self.nlg.train(target_seq, target_lens, db, da, bs, no_prop=True)

    # Train E2E
    input_seq, input_lens, target_seq, target_lens, db, bs = self.e2e.prep_batch(batch_rows)
    e2e_loss = self.e2e.train(input_seq, input_lens, target_seq, target_lens, db, bs, no_prop=True)

    # Combine loss and backprop
    comb_loss = nlu_loss + dm_loss + nlg_loss + e2e_loss

    comb_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return comb_loss.item()

  def save(self, name):
    torch.save(self, name+'.mt')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.mt').state_dict())

  def prep_batch(self, batch_rows):
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)
    input_seq, input_lens, target_seq, target_lens, db, bs = self.e2e.prep_batch(batch_rows)
    return input_seq, input_lens, target_seq, target_lens, db, bs, da

  def decode(self, input_seq, input_lens, max_len, db, bs, da):
    return self.e2e.decode(input_seq, input_lens, max_len, db, bs)

class MultiTaskFusion(nn.Module):
  def __init__(self, nlu, dm, nlg, encoder, pnn, decoder, cf_dec, input_w2i, output_w2i, args):
    super(MultiTaskFusion, self).__init__()

    # Model
    self.nlu = nlu
    self.dm = dm
    self.nlg = nlg

    self.encoder = encoder
    self.pnn = pnn
    self.decoder = decoder

    self.f_enc = nn.Linear(args.hid_size+args.bs_size, args.hid_size)
    self.f_pnn = nn.Linear(args.hid_size+args.da_size+args.bs_size, args.hid_size)
    self.f_dec = cf_dec

    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.args = args

  def train(self, batch_rows):
    self.optim.zero_grad()

    # Get values
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)

    # Train E2E
    proba = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs, None)
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
    
    # Backprop
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs, da):
    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

    # NLU output
    pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).unsqueeze(0)

    # Encoder Fusion
    enc_hid = self.f_enc(torch.cat((encoder_hidden[0], pred_bs), dim=-1))
     
    # PNN
    pnn_out = self.pnn(enc_hid, db)

    # DM output
    pred_da = F.sigmoid(self.dm(pred_bs, db))

    # Policy Fusion
    pnn_out = self.f_pnn(torch.cat((pnn_out, pred_da, pred_bs), dim=-1))
    decoder_hidden = (pnn_out, encoder_hidden[1])

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w)).cuda()
    last_word = target_seq[0].unsqueeze(0)

    # NLG 
    da_proj = self.nlg.da_in(pred_da)
    db_proj = self.nlg.db_in(db)
    bs_proj = self.nlg.bs_in(pred_bs)
    da_projc = self.nlg.da_inc(pred_da)
    db_projc = self.nlg.db_inc(db)
    bs_projc = self.nlg.bs_inc(pred_bs)
    nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

    for t in range(1,target_seq.size(0)):
      decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)
      nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

      # Save output
      probas[t] = self.f_dec(dec_out, nlg_decoder_output)

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas

  def prep_batch(self, batch_rows):
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)
    return input_seq, input_lens, target_seq, target_lens, db, bs, da

  def decode(self, input_seq, input_lens, max_len, db, bs, da):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    # Should not be used.
    da = None

    with torch.no_grad():
      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens)

      # NLU output
      pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).detach().unsqueeze(0)

      # Encoder Fusion
      enc_hid = self.f_enc(torch.cat((encoder_hidden[0], pred_bs), dim=-1))
       
      # PNN
      pnn_out = self.pnn(enc_hid, db)

      # DM output
      pred_da = F.sigmoid(self.dm(pred_bs, db)).detach()

      # Policy Fusion
      pnn_out = self.f_pnn(torch.cat((pnn_out, pred_da, pred_bs), dim=-1))
      decoder_hidden = (pnn_out, encoder_hidden[1])

      # Decoder
      last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      # NLG
      da_proj = self.nlg.da_in(pred_da)
      db_proj = self.nlg.db_in(db)
      bs_proj = self.nlg.bs_in(pred_bs)
      da_projc = self.nlg.da_inc(pred_da)
      db_projc = self.nlg.db_inc(db)
      bs_projc = self.nlg.bs_inc(pred_bs)
      nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

      for t in range(max_len):
        nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

        # Pass through decoder
        decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)

        decoder_output = self.f_dec(dec_out, nlg_decoder_output)

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences
  
  def save(self, name):
    torch.save(self, name+'.mtfusion')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.mtfusion').state_dict())

class StructuredFusion(nn.Module):
  def __init__(self, nlu, dm, nlg, encoder, pnn, decoder, cf_dec, input_w2i, output_w2i, args):
    super(StructuredFusion, self).__init__()

    # Model
    self.nlu = nlu
    self.dm = dm
    self.nlg = nlg

    self.encoder = encoder
    self.pnn = pnn
    self.decoder = decoder

    self.f_dec = cf_dec

    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

    #sl_params = list(self.nlu.parameters()) + list(self.nlg.parameters()) + list(self.dm.parameters()) #+ list(self.f_dec.parameters())
    #rl_params = list(self.encoder.parameters()) + list(self.pnn.parameters()) + list(self.decoder.parameters()) + list(self.f_dec.parameters())
    #self.sl_optim = optim.Adam(lr=args.lr, params=sl_params, weight_decay=args.l2_norm)
    #self.rl_optim = optim.Adam(lr=args.lr, params=rl_params, weight_decay=args.l2_norm)

    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.bs_criterion = nn.BCELoss(reduce=True)
    self.da_criterion = nn.BCELoss(reduce=True)
    self.args = args

  def train(self, batch_rows):
    self.optim.zero_grad()

    # Get values
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)

    # Train E2E
    proba, pred_bs, pred_da = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs, None)
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
    if self.args.multitasking:
        loss += self.bs_criterion(pred_bs, bs)
        loss += self.da_criterion(pred_da, da)

    # Backprop
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs, da):
    # NLU output
    pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).unsqueeze(0)
    if self.args.tune_params is False:
        pred_bs = pred_bs.detach()
    #bs_inp = self.bs_comb(torch.cat((pred_bs,bs.unsqueeze(0)), dim=-1))

    bs_inp = bs.unsqueeze(0)
    #bs_inp = pred_bs + bs.unsqueeze(0)

    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens, bs_inp)

    # DM output
    pred_da = F.sigmoid(self.dm(bs_inp, db))
    if self.args.tune_params is False:
        pred_da = pred_da.detach()

    # PNN
    pnn_out = self.pnn(encoder_hidden[0], db, bs_inp, pred_da)
    decoder_hidden = (pnn_out, encoder_hidden[1])

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w)).cuda()
    last_word = target_seq[0].unsqueeze(0)

    # NLG 
    da_proj = self.nlg.da_in(pred_da)
    db_proj = self.nlg.db_in(db)
    bs_proj = self.nlg.bs_in(bs_inp)
    da_projc = self.nlg.da_inc(pred_da)
    db_projc = self.nlg.db_inc(db)
    bs_projc = self.nlg.bs_inc(bs_inp)
    nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

    for t in range(1,target_seq.size(0)):
      decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)
      nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

      if self.args.tune_params is False:
        self.nlg_decoder_output = self.nlg_decoder_output.detach()
      # Save output
      probas[t] = F.log_softmax(self.f_dec(dec_out, nlg_decoder_output), dim=-1)

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas, pred_bs, pred_da

  def prep_batch(self, batch_rows):
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)
    return input_seq, input_lens, target_seq, target_lens, db, bs, da

  def rl_forward(self, input_seq, input_lens, max_len, db, bs, da):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len)).cuda()
    logprobas = torch.zeros((batch_size, max_len)).cuda()

    # Should not be used.
    da = None

    # NLU output
    pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).unsqueeze(0)
    # pred_bs2 = torch.cat((pred_bs,bs.unsqueeze(0)), dim=-1)
    pred_bs = bs.unsqueeze(0)

    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens, pred_bs)

    # DM output
    pred_da = F.sigmoid(self.dm(pred_bs, db)).detach()

    # PNN
    pnn_out = self.pnn(encoder_hidden[0], db, pred_bs, pred_da)
    decoder_hidden = (pnn_out, encoder_hidden[1])

    # Decoder
    last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

    # NLG
    da_proj = self.nlg.da_in(pred_da)
    db_proj = self.nlg.db_in(db)
    bs_proj = self.nlg.bs_in(pred_bs)
    da_projc = self.nlg.da_inc(pred_da)
    db_projc = self.nlg.db_inc(db)
    bs_projc = self.nlg.bs_inc(pred_bs)
    nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

    def _sample(dec_output, temp=0.05):
        # dec_output: (1, 1, vocab_size), need to softmax and log_softmax
        dec_output = dec_output.view(batch_size, -1) # (batch_size, vocab_size, )
        # Mask out _UNK, _PAD 
        inds = torch.arange(dec_output.size(1)).unsqueeze(0).repeat(dec_output.size(0), 1)
        mask = - 1e5 * ((inds == 2) + (inds == 3)).float().cuda()
        prob = F.softmax((dec_output + mask)/temp, dim=1) # (batch_size, vocab_size, )
        logprob = F.log_softmax(dec_output, dim=1) # (batch_size, vocab_size, )
        symbol = prob.multinomial(num_samples=1).detach() # (batch_size, 1)
        # _, symbol = prob.topk(1) # (1, )
        # print('multinomial symbol = {}, prob = {}'.format(symbol, prob[symbol.item()]))
        # print('topk symbol = {}, prob = {}'.format(tmp_symbol, prob[tmp_symbol.item()]))
        logprob = logprob.gather(1, symbol) # (1, )
        return logprob, symbol

    for t in range(max_len):
      nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

      # Pass through decoder
      decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)

      decoder_output = self.f_dec(dec_out, nlg_decoder_output.detach())

      # Get top candidates
      #topv, topi = decoder_output.topk(1)
      topv, topi = _sample(decoder_output)

      predictions[:, t] = topi.view(-1)
      logprobas[:, t] = topv.view(-1)

      # Set new last word
      last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    #print("TODO: sample")
    return logprobas, predicted_sentences

  def decode(self, input_seq, input_lens, max_len, db, bs, da):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    # Should not be used.
    da = None

    with torch.no_grad():
      # NLU output
      pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).unsqueeze(0).detach()
      # pred_bs2 = torch.cat((pred_bs,bs.unsqueeze(0)), dim=-1)
      pred_bs = bs.unsqueeze(0)
      #pred_bs = pred_bs + bs.unsqueeze(0)
      #pred_bs = torch.cat((pred_bs,bs.unsqueeze(0)), dim=-1)

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens, pred_bs)

      # DM output
      pred_da = F.sigmoid(self.dm(pred_bs, db)).detach()

      # PNN
      pnn_out = self.pnn(encoder_hidden[0], db, pred_bs, pred_da)
      decoder_hidden = (pnn_out, encoder_hidden[1])

      # Decoder
      last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      # NLG
      da_proj = self.nlg.da_in(pred_da)
      db_proj = self.nlg.db_in(db)
      bs_proj = self.nlg.bs_in(pred_bs)
      da_projc = self.nlg.da_inc(pred_da)
      db_projc = self.nlg.db_inc(db)
      bs_projc = self.nlg.bs_inc(pred_bs)
      nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

      for t in range(max_len):
        nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

        # Pass through decoder
        decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)

        decoder_output = self.f_dec(dec_out, nlg_decoder_output)

        inds = torch.arange(decoder_output.size(2)).unsqueeze(0).repeat(decoder_output.size(1), 1).unsqueeze(0)
        mask = - 1e5 * ((inds == 2) + (inds == 3)).float().cuda()
        decoder_output = F.softmax(decoder_output + mask, dim=-1)

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences
  
  def save(self, name):
    torch.save(self, name+'.mtfusion')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.mtfusion').state_dict())

############################################################### LEGACY ################################################################

class MultiTaskFusion2(nn.Module):
  def __init__(self, nlu, dm, nlg, encoder, pnn, decoder, cf_dec, input_w2i, output_w2i, args):
    super(MultiTaskFusion2, self).__init__()

    # Model
    self.nlu = nlu
    self.dm = dm
    self.nlg = nlg

    self.encoder = encoder
    self.pnn = pnn
    self.decoder = decoder

    self.f_dec = cf_dec

    self.input_i2w = sorted(input_w2i, key=input_w2i.get)
    self.input_w2i = input_w2i
    self.output_i2w = sorted(output_w2i, key=output_w2i.get)
    self.output_w2i = output_w2i

    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)
    self.criterion = nn.NLLLoss(ignore_index=3, size_average=True)
    self.bs_criterion = nn.BCELoss(reduce=True)
    self.da_criterion = nn.BCELoss(reduce=True)
    self.args = args

  def train(self, batch_rows):
    self.optim.zero_grad()

    # Get values
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)

    # Train E2E
    proba, pred_bs, pred_da = self.forward(input_seq, input_lens, target_seq, target_lens, db, bs, None)
    loss = self.criterion(proba.view(-1, proba.size(-1)), target_seq.flatten())
    if self.args.multitasking:
        loss += self.bs_criterion(pred_bs, bs)
        loss += self.da_criterion(pred_da, da)

    # Backprop
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def forward(self, input_seq, input_lens, target_seq, target_lens, db, bs, da):
    # NLU output
    pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).unsqueeze(0)
    if self.args.tune_params is False:
        pred_bs = pred_bs.detach()
    #pred_bs2 = torch.cat((pred_bs,bs.unsqueeze(0)), dim=-1)

    bs_inp = bs.unsqueeze(0)
    # bs_inp = pred_bs

    # Encoder
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens, bs_inp)

    # DM output
    pred_da = F.sigmoid(self.dm(bs_inp, db))
    if self.args.tune_params is False:
        pred_da = pred_da.detach()

    # PNN
    pnn_out = self.pnn(encoder_hidden[0], db, bs_inp, pred_da)
    decoder_hidden = (pnn_out, encoder_hidden[1])

    # Decoder
    probas = torch.zeros(target_seq.size(0), target_seq.size(1), len(self.output_i2w)).cuda()
    last_word = target_seq[0].unsqueeze(0)

    # NLG 
    da_proj = self.nlg.da_in(pred_da)
    db_proj = self.nlg.db_in(db)
    bs_proj = self.nlg.bs_in(bs_inp)
    da_projc = self.nlg.da_inc(pred_da)
    db_projc = self.nlg.db_inc(db)
    bs_projc = self.nlg.bs_inc(bs_inp)
    nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

    for t in range(1,target_seq.size(0)):
      decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)
      nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

      if self.args.tune_params is False:
        self.nlg_decoder_output = self.nlg_decoder_output.detach()
      # Save output
      probas[t] = self.f_dec(dec_out, nlg_decoder_output)

      # Set new last word
      last_word = target_seq[t].unsqueeze(0)

    return probas, pred_bs, pred_da

  def prep_batch(self, batch_rows):
    input_seq, input_lens, bs = self.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.nlg.prep_batch(batch_rows)
    return input_seq, input_lens, target_seq, target_lens, db, bs, da

  def decode(self, input_seq, input_lens, max_len, db, bs, da):
    batch_size = len(input_seq)
    predictions = torch.zeros((batch_size, max_len))

    # Should not be used.
    da = None

    with torch.no_grad():
      # NLU output
      pred_bs = F.sigmoid(self.nlu(input_seq, input_lens)).unsqueeze(0).detach()
      # pred_bs2 = torch.cat((pred_bs,bs.unsqueeze(0)), dim=-1)
      pred_bs = bs.unsqueeze(0)

      # Encoder
      encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lens, pred_bs)

      # DM output
      pred_da = F.sigmoid(self.dm(pred_bs, db)).detach()

      # PNN
      pnn_out = self.pnn(encoder_hidden[0], db, pred_bs, pred_da)
      decoder_hidden = (pnn_out, encoder_hidden[1])

      # Decoder
      last_word = torch.cuda.LongTensor([[self.output_w2i['_GO'] for _ in range(len(input_seq))]])

      # NLG
      da_proj = self.nlg.da_in(pred_da)
      db_proj = self.nlg.db_in(db)
      bs_proj = self.nlg.bs_in(pred_bs)
      da_projc = self.nlg.da_inc(pred_da)
      db_projc = self.nlg.db_inc(db)
      bs_projc = self.nlg.bs_inc(pred_bs)
      nlg_decoder_hidden = (F.relu(da_proj+db_proj+bs_proj), F.relu(da_projc+db_projc+bs_projc))

      for t in range(max_len):
        nlg_decoder_output, nlg_decoder_hidden = self.nlg.decoder(nlg_decoder_hidden, last_word, None, None)

        # Pass through decoder
        decoder_output, decoder_hidden, dec_out = self.decoder(decoder_hidden, last_word, None, None, ret_out=True)

        decoder_output = self.f_dec(dec_out, nlg_decoder_output)

        # Get top candidates
        topv, topi = decoder_output.data.topk(1)
        topi = topi.view(-1)

        predictions[:, t] = topi

        # Set new last word
        last_word = topi.detach().view(1, -1)

    predicted_sentences = []
    for sentence in predictions:
      sent = []
      for ind in sentence:
        word = self.output_i2w[ind.long().item()]
        if word == '_EOS':
          break
        sent.append(word)
      predicted_sentences.append(' '.join(sent))

    return predicted_sentences
  
  def save(self, name):
    torch.save(self, name+'.mtfusion')

  def load(self, name):
    self.load_state_dict(torch.load(name+'.mtfusion').state_dict())

class RLAgent(object):
  def __init__(self, model, evaluator):
    self.model = model
    self.evaluator = evaluator
    self.target = json.load(open('data/train_dials.json'))
    self.all_rewards = []

  def rl_train(self, batch_rows):
    self.model.optim.zero_grad()

    # Get values
    input_seq, input_lens, bs = self.model.nlu.prep_batch(batch_rows)
    target_seq, target_lens, db, da, bs = self.model.nlg.prep_batch(batch_rows)

    # RL forward
    lps, sents = self.model.rl_forward(input_seq, input_lens, max(target_lens), db, bs, None)
    if random.random() < 0.01:
      print(sents)
    #print(sents)

    # Evaluate reward
    pred = defaultdict(list)
    inds = defaultdict(list)
    for i in range(len(sents)):
      pred[batch_rows[i][-2]].append(sents[i])
      inds[batch_rows[i][-2]].append(batch_rows[i][-1])

    match, success = self.evaluator.evaluateModel(pred, inds, self.target, mode='test')
    reward_dict = {p:s for p,s,m in zip(pred.keys(),success,match) }
    reward = [reward_dict[batch_rows[i][-2]] for i in range(len(sents))]

    # normalize rewards
    self.all_rewards += reward
    reward = [(r - np.mean(self.all_rewards))/max(1e-4, np.std(self.all_rewards)) for r in reward]

    # Apply discounting
    #reward = torch.cuda.FloatTensor([reward]).view(1,1).repeat(*lps.size()).float()
    reward = torch.cuda.FloatTensor(reward).unsqueeze(1).repeat(1, lps.size(1)).float()  
    discount = 0.99**torch.arange(lps.size(1)).unsqueeze(0).repeat(lps.size(0),1).float().cuda()
    lens = [len(s.split()) + 1 for s in sents]
    mask = (torch.arange(lps.size(1)).unsqueeze(0).repeat(lps.size(0),1).cuda() < torch.cuda.LongTensor(lens).unsqueeze(1).repeat(1, lps.size(1))).float()

    loss = -(reward * discount * mask * lps).sum()
    #print(success, match, loss)

    # Backprop
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.args.clip)
    self.model.rl_optim.step()

    return loss.item()
