import copy
import os
import json
import random
import sys
import sqlite3

from delexicalize import normalize


sys.path.append('..')

class MWEval(object):
  def __init__(self):
      domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
      self.dbs = {}
      CUR_DIR = os.path.dirname(__file__).replace('latent_dialog', '')
      
      for domain in domains:
          db = os.path.join(CUR_DIR, 'db/{}-dbase.db'.format(domain))
          conn = sqlite3.connect(db)
          c = conn.cursor()
          self.dbs[domain] = c

      fin1 = open('data/multi-woz/delex.json')
      self.delex_dialogues = json.load(fin1)
    
  def queryResultVenues(self, domain, turn, real_belief=False):
      # query the db
      sql_query = "select * from {}".format(domain)
  
      if real_belief == True:
          items = turn.items()
      else:
          items = turn['metadata'][domain]['semi'].items()
  
      flag = True
      for key, val in items:
          if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
              pass
          else:
              if flag:
                  sql_query += " where "
                  val2 = val.replace("'", "''")
                  val2 = normalize(val2)
                  if key == 'leaveAt':
                      sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                  elif key == 'arriveBy':
                      sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                  else:
                      sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                  flag = False
              else:
                  val2 = val.replace("'", "''")
                  val2 = normalize(val2)
                  if key == 'leaveAt':
                      sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                  elif key == 'arriveBy':
                      sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                  else:
                      sql_query += r" and " + key + "=" + r"'" + val2 + r"'"
  
      try:  # "select * from attraction  where name = 'queens college'"
          return self.dbs[domain].execute(sql_query).fetchall()
      except:
          return []  # TODO test it


  def evaluateModel(self, dialogues, inds, val_dials, mode='valid'):
      """Gathers statistics for the whole sets."""
      delex_dialogues = self.delex_dialogues
      successes, matches = [], []
      total = 0
  
      gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
               'hospital': [0, 0, 0], 'police': [0, 0, 0]}
      sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                       'taxi': [0, 0, 0],
                       'hospital': [0, 0, 0], 'police': [0, 0, 0]}
  
      for filename, dial in dialogues.items():
          data = copy.copy(delex_dialogues[filename])
          ind = []
          for i in inds[filename]:
            ind.append(2*i)
            ind.append(2*i+1)
          data['log'] = [data['log'][i] for i in ind]
  
          goal, _, _, requestables, _ = self.evaluateRealDialogue(data, filename)
  
          success, match, stats = self.evaluateGeneratedDialogue(dial, goal, data, requestables)
  
          successes.append(success)
          matches.append(match)
          total += 1
  
          for domain in gen_stats.keys():
              gen_stats[domain][0] += stats[domain][0]
              gen_stats[domain][1] += stats[domain][1]
              gen_stats[domain][2] += stats[domain][2]
  
          if 'SNG' in filename:
              for domain in gen_stats.keys():
                  sng_gen_stats[domain][0] += stats[domain][0]
                  sng_gen_stats[domain][1] += stats[domain][1]
                  sng_gen_stats[domain][2] += stats[domain][2]
  
      return matches, successes 
  
  def parseGoal(self, goal, d, domain):
      """Parses user goal into dictionary format."""
      goal[domain] = {}
      goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
      if 'info' in d['goal'][domain]:
          if domain == 'train':
              # we consider dialogues only where train had to be booked!
              if 'book' in d['goal'][domain]:
                  goal[domain]['requestable'].append('reference')
              if 'reqt' in d['goal'][domain]:
                  if 'trainID' in d['goal'][domain]['reqt']:
                      goal[domain]['requestable'].append('id')
          else:
              if 'reqt' in d['goal'][domain]:
                  for s in d['goal'][domain]['reqt']:  # addtional requests:
                      if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                          # ones that can be easily delexicalized
                          goal[domain]['requestable'].append(s)
              if 'book' in d['goal'][domain]:
                  goal[domain]['requestable'].append("reference")
  
          goal[domain]["informable"] = d['goal'][domain]['info']
          if 'book' in d['goal'][domain]:
              goal[domain]["booking"] = d['goal'][domain]['book']
  
      return goal
  
  def evaluateGeneratedDialogue(self, dialog, goal, realDialogue, real_requestables):
      """Evaluates the dialogue created by the model.
      First we load the user goal of the dialogue, then for each turn
      generated by the system we look for key-words.
      For the Inform rate we look whether the entity was proposed.
      For the Success rate we look for requestables slots"""
      # for computing corpus success
      requestables = ['phone', 'address', 'postcode', 'reference', 'id']
  
      # CHECK IF MATCH HAPPENED
      provided_requestables = {}
      venue_offered = {}
      domains_in_goal = []
  
      for domain in goal.keys():
          venue_offered[domain] = []
          provided_requestables[domain] = []
          domains_in_goal.append(domain)
  
      for t, sent_t in enumerate(dialog):
          for domain in goal.keys():
              # for computing success
              if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                  if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                      # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                      venues = self.queryResultVenues(domain, realDialogue['log'][t*2 + 1])
  
                      # if venue has changed
                      if len(venue_offered[domain]) == 0 and venues:
                          venue_offered[domain] = random.sample(venues, 1)
                      else:
                          flag = False
                          for ven in venues:
                              if venue_offered[domain][0] == ven:
                                  flag = True
                                  break
                          if not flag and venues:  # sometimes there are no results so sample won't work
                              # print venues
                              venue_offered[domain] = random.sample(venues, 1)
                  else:  # not limited so we can provide one
                      venue_offered[domain] = '[' + domain + '_name]'
  
              # ATTENTION: assumption here - we didn't provide phone or address twice! etc
              for requestable in requestables:
                  if requestable == 'reference':
                      if domain + '_reference' in sent_t:
                          if 'restaurant_reference' in sent_t:
                              if realDialogue['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                  provided_requestables[domain].append('reference')
  
                          elif 'hotel_reference' in sent_t:
                              if realDialogue['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                  provided_requestables[domain].append('reference')
  
                          elif 'train_reference' in sent_t:
                              if realDialogue['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                  provided_requestables[domain].append('reference')
  
                          else:
                              provided_requestables[domain].append('reference')
                  else:
                      if domain + '_' + requestable + ']' in sent_t:
                          provided_requestables[domain].append(requestable)
  
      # if name was given in the task
      for domain in goal.keys():
          # if name was provided for the user, the match is being done automatically
          if 'info' in realDialogue['goal'][domain]:
              if 'name' in realDialogue['goal'][domain]['info']:
                  venue_offered[domain] = '[' + domain + '_name]'
  
          # special domains - entity does not need to be provided
          if domain in ['taxi', 'police', 'hospital']:
              venue_offered[domain] = '[' + domain + '_name]'
  
  
          if domain == 'train':
              if not venue_offered[domain]:
                  if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
                      venue_offered[domain] = '[' + domain + '_name]'
  
      """
      Given all inform and requestable slots
      we go through each domain from the user goal
      and check whether right entity was provided and
      all requestable slots were given to the user.
      The dialogue is successful if that's the case for all domains.
      """
      # HARD EVAL
      stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
               'hospital': [0, 0, 0], 'police': [0, 0, 0]}
  
      match = 0
      success = 0
      # MATCH
      for domain in goal.keys():
          match_stat = 0
          if domain in ['restaurant', 'hotel', 'attraction', 'train']:
              goal_venues = self.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
              if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                  match += 1
                  match_stat = 1
              elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                  match += 1
                  match_stat = 1
          else:
              if domain + '_name]' in venue_offered[domain]:
                  match += 1
                  match_stat = 1
  
          stats[domain][0] = match_stat
          stats[domain][2] = 1
  
      if match == len(goal.keys()):
          match = 1
      else:
          match = 0
  
      # SUCCESS
      if match:
          for domain in domains_in_goal:
              success_stat = 0
              domain_success = 0
              if len(real_requestables[domain]) == 0:
                  success += 1
                  success_stat = 1
                  stats[domain][1] = success_stat
                  continue
              # if values in sentences are super set of requestables
              for request in set(provided_requestables[domain]):
                  if request in real_requestables[domain]:
                      domain_success += 1
  
              if domain_success >= len(real_requestables[domain]):
                  success += 1
                  success_stat = 1
  
              stats[domain][1] = success_stat
  
          # final eval
          if success >= len(real_requestables):
              success = 1
          else:
              success = 0
  
      #rint requests, 'DIFF', requests_real, 'SUCC', success
      return success, match, stats
  
  
  def evaluateRealDialogue(self, dialog, filename):
      """Evaluation of the real dialogue.
      First we loads the user goal and then go through the dialogue history.
      Similar to evaluateGeneratedDialogue above."""
      domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
      requestables = ['phone', 'address', 'postcode', 'reference', 'id']
  
      # get the list of domains in the goal
      domains_in_goal = []
      goal = {}
      for domain in domains:
          if dialog['goal'][domain]:
              goal = self.parseGoal(goal, dialog, domain)
              domains_in_goal.append(domain)
  
      # compute corpus success
      real_requestables = {}
      provided_requestables = {}
      venue_offered = {}
      for domain in goal.keys():
          provided_requestables[domain] = []
          venue_offered[domain] = []
          real_requestables[domain] = goal[domain]['requestable']
  
      # iterate each turn
      m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
      for t in range(len(m_targetutt)):
          for domain in domains_in_goal:
              sent_t = m_targetutt[t]
              # for computing match - where there are limited entities
              if domain + '_name' in sent_t or '_id' in sent_t:
                  if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                      # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                      venues = self.queryResultVenues(domain, dialog['log'][t * 2 + 1])
  
                      # if venue has changed
                      if len(venue_offered[domain]) == 0 and venues:
                          venue_offered[domain] = random.sample(venues, 1)
                      else:
                          flag = False
                          for ven in venues:
                              if venue_offered[domain][0] == ven:
                                  flag = True
                                  break
                          if not flag and venues:  # sometimes there are no results so sample won't work
                              #print venues
                              venue_offered[domain] = random.sample(venues, 1)
                  else:  # not limited so we can provide one
                      venue_offered[domain] = '[' + domain + '_name]'
  
              for requestable in requestables:
                  # check if reference could be issued
                  if requestable == 'reference':
                      if domain + '_reference' in sent_t:
                          if 'restaurant_reference' in sent_t:
                              if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                  provided_requestables[domain].append('reference')
  
                          elif 'hotel_reference' in sent_t:
                              if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                  provided_requestables[domain].append('reference')
  
                                  #return goal, 0, match, real_requestables
                          elif 'train_reference' in sent_t:
                              if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                  provided_requestables[domain].append('reference')
  
                          else:
                              provided_requestables[domain].append('reference')
                  else:
                      if domain + '_' + requestable in sent_t:
                          provided_requestables[domain].append(requestable)
  
      # offer was made?
      for domain in domains_in_goal:
          # if name was provided for the user, the match is being done automatically
          if 'info' in dialog['goal'][domain]:
              if 'name' in dialog['goal'][domain]['info']:
                  venue_offered[domain] = '[' + domain + '_name]'
  
          # special domains - entity does not need to be provided
          if domain in ['taxi', 'police', 'hospital']:
              venue_offered[domain] = '[' + domain + '_name]'
  
          # if id was not requested but train was found we dont want to override it to check if we booked the right train
          if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
              venue_offered[domain] = '[' + domain + '_name]'
  
      # HARD (0-1) EVAL
      stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
               'hospital': [0, 0, 0], 'police': [0, 0, 0]}
  
      match, success = 0, 0
      # MATCH
      for domain in goal.keys():
          match_stat = 0
          if domain in ['restaurant', 'hotel', 'attraction', 'train']:
              goal_venues = self.queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
              #print(goal_venues)
              if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                  match += 1
                  match_stat = 1
              elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                  match += 1
                  match_stat = 1
  
          else:
              if domain + '_name' in venue_offered[domain]:
                  match += 1
                  match_stat = 1
  
          stats[domain][0] = match_stat
          stats[domain][2] = 1
  
      if match == len(goal.keys()):
          match = 1
      else:
          match = 0
  
      # SUCCESS
      if match:
          for domain in domains_in_goal:
              domain_success = 0
              success_stat = 0
              if len(real_requestables[domain]) == 0:
                  # check that
                  success += 1
                  success_stat = 1
                  stats[domain][1] = success_stat
                  continue
              # if values in sentences are super set of requestables
              for request in set(provided_requestables[domain]):
                  if request in real_requestables[domain]:
                      domain_success += 1
  
              if domain_success >= len(real_requestables[domain]):
                  success +=1
                  success_stat = 1
  
              stats[domain][1] = success_stat
  
          # final eval
          if success >= len(real_requestables):
              success = 1
          else:
              success = 0
  
      return goal, success, match, real_requestables, stats
