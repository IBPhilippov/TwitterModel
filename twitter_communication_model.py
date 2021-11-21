import json
import networkx as nx
import pandas as pd
import numpy as np
import random
import igraph as ig
from math import sqrt,log
import argparse
import datetime
from time import sleep
from numpy.random import RandomState


rnseed=1
param_b = 0.15
param_regime = 1
bots_regime=15

experiment_mode=True 
experiment_path='YOUR PATH FOR NETWORK SNAPSHOTS'
experiment_graduality=100 #snapshot frequency
experiment_parameters={'value_regime': [0,1,2,3,4,5,6,7], 'b':np.arange(0,0.51,0.05), 'botshareFirst':[0.5,1],
                      'bots_regime': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                      }

leaders_positions_list=[[-0.5,0],[-0.5,0.5]]

bots_regime_dict={0: {'bots_retweet': 0,
  'bots_write_tweets': 0,
  'necroposting': 0,
  'spam_following': 0},
 1: {'bots_retweet': 1,
  'bots_write_tweets': 0,
  'necroposting': 0,
  'spam_following': 0},
 2: {'bots_retweet': 0,
  'bots_write_tweets': 1,
  'necroposting': 0,
  'spam_following': 0},
 3: {'bots_retweet': 1,
  'bots_write_tweets': 1,
  'necroposting': 0,
  'spam_following': 0},
 4: {'bots_retweet': 0,
  'bots_write_tweets': 0,
  'necroposting': 1,
  'spam_following': 0},
 5: {'bots_retweet': 1,
  'bots_write_tweets': 0,
  'necroposting': 1,
  'spam_following': 0},
 6: {'bots_retweet': 0,
  'bots_write_tweets': 1,
  'necroposting': 1,
  'spam_following': 0},
 7: {'bots_retweet': 1,
  'bots_write_tweets': 1,
  'necroposting': 1,
  'spam_following': 0},
 8: {'bots_retweet': 0,
  'bots_write_tweets': 0,
  'necroposting': 0,
  'spam_following': 1},
 9: {'bots_retweet': 1,
  'bots_write_tweets': 0,
  'necroposting': 0,
  'spam_following': 1},
 10: {'bots_retweet': 0,
  'bots_write_tweets': 1,
  'necroposting': 0,
  'spam_following': 1},
 11: {'bots_retweet': 1,
  'bots_write_tweets': 1,
  'necroposting': 0,
  'spam_following': 1},
 12: {'bots_retweet': 0,
  'bots_write_tweets': 0,
  'necroposting': 1,
  'spam_following': 1},
 13: {'bots_retweet': 1,
  'bots_write_tweets': 0,
  'necroposting': 1,
  'spam_following': 1},
 14: {'bots_retweet': 0,
  'bots_write_tweets': 1,
  'necroposting': 1,
  'spam_following': 1},
 15: {'bots_retweet': 1,
  'bots_write_tweets': 1,
  'necroposting': 1,
  'spam_following': 1}}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): 
            return None
        return json.JSONEncoder.default(self, obj)

def get_defaults(): 
    alpha=1.6 
    numleaders=2
    max_activity=10
    alpha2=1.2 
    lifespan=5 
    leader_boost=0.2 
    subscription_probability=0.5
    totalP=1000 # total population
    b=0.1 #fraction of bots
    tolerance=0.5
    steps=500 
    botshareFirst=1 #share of bots attached to 1st leader
    l_a_b=True #leaders attached to bots

    followings_reaction=1
    botsfollowings=1
    reciprocity_probability=0.1
    bot_subscriptions_limit=100
    return alpha,numleaders,max_activity,alpha2,lifespan,leader_boost,subscription_probability,totalP,b,tolerance,steps,botshareFirst,l_a_b,followings_reaction,botsfollowings,reciprocity_probability,bot_subscriptions_limit

def changed_eges(G,G_s):
    newedges=0
    removededges=0
    for edge in G.edges():
        if edge not in G_s.edges():
            newedges+=1
    for edge in G_s.edges():
        if edge not in G.edges():
            removededges+=1
    return newedges,removededges

def generate_network(P): 
    global prng
    g=ig.Graph.Barabasi(n=P,m=4,directed=True)
    G=nx.DiGraph()
    G.add_nodes_from(range(0, P))
    G.add_edges_from([(x[0], x[1]) for x in g.get_edgelist()])
    for i in G.nodes():
        G.nodes[i]['bot']=0
        G.nodes[i]['leader']=0
        G.nodes[i]['position']=None
        cursor=0
        while G.out_degree()[i]<4:
            target=sorted(G.in_degree(), key=lambda node: node[1],reverse=True)[cursor][0]
            if target!=i and target not in [n for n in G.neighbors(i)]:
                G.add_edges_from([(i, target) for x in g.get_edgelist()])
            else:
                cursor+=1
    G,leaderlist=determine_leaders(G)
    G=determine_activity_and_pc(G,alpha,max_activity,alpha2)
    G=determine_positions(G,leaderlist)
    G,bots=add_bots(G,B,leaderlist)
    for i in G.nodes():
        G.nodes[i]['feed']=[]
        G.nodes[i]['clock']=0
        G.nodes[i]['influenced']=0
        G.nodes[i]['tweets_read']=0
        G.nodes[i]['unreacted_followings']=[]
        G.nodes[i]['subscription_actions']=0
        G.nodes[i]['influenced_retweets']=0
        G.nodes[i]['retweets_made']=0
    
    return G,bots



def determine_leaders(G): 
    global prng
    global leaderposts
    leaders=sorted(G.in_degree, key=lambda x: x[1], reverse=True)[0:numleaders]
    leaderlist=[x[0] for x in leaders]
    for i in leaderlist:
        G.nodes[i]['leader']=1
        leaderposts[i]=[]
    return G,leaderlist



def determine_positions(G,leaderlist):
    global prng
    global leaders_positions
    num_gen=0
    leaders_positions2=[x for x in leaders_positions]
    for leader in leaderlist:
        G.nodes[leader]['position']=prng.choice(leaders_positions2)
        leaders_positions2.remove(G.nodes[leader]['position'])
        num_gen+=1
    for i in G.nodes():
        if G.nodes[i]['position'] is None:
            determined_neighbours=0
            for n in G[i]:
                if G.nodes[n]['position'] is not None:
                    determined_neighbours+=1
            if determined_neighbours==0:
                G.nodes[i]['position']=prng.beta(2, 2)*2-1
                num_gen+=1
    while num_gen<P:
        position=prng.beta(2, 2)*2-1
        generated=0
        i=0
        while i<P and generated==0:
            if G.nodes[i]['position'] is None:
                neighbours_pos=[G.nodes[n]['position'] for n in G[i]]
                for a in range(0,len(neighbours_pos)):
                    try:
                        neighbours_pos.remove(None)
                    except:
                        pass
                mean=np.mean(neighbours_pos)
                if mean-position<tolerance and mean-position>-tolerance:
                    G.nodes[i]['position']=position
                    generated=1
                    num_gen+=1
            i+=1
    return G



def determine_activity_and_pc(G,alpha, max_activity,alpha2):
    global prng
    for i in G.nodes():
        if  G.nodes[i]['leader']==0:
            activity=1+round(prng.pareto(alpha))
            if activity>max_activity:
                activity=max_activity
            G.nodes[i]['activity']=max_activity-activity+1
            pc=4+round(prng.pareto(alpha2))
            G.nodes[i]['pc']=pc
        else:
            activity=round(prng.uniform(max_activity-4,max_activity))
            G.nodes[i]['activity']=max_activity-activity+1
            pc=15+round(prng.pareto(alpha2))
            G.nodes[i]['pc']=pc
    return G



def add_bots(G,B,leaderlist):
    global P
    global prng
    global botshare
    botdict={}
    numlead=0
    leaders_position=[G.nodes()[x]['position'] for x in leaderlist]
    leaderlist=[x for _, x in sorted(list(zip(leaders_position, leaderlist)), key=lambda x: x[0], reverse=False)]
    l=leaderlist[numlead]
    for leader in leaderlist:
        botdict[leader]=[]
    for i in range(0,B):
        G.add_node(P+i)
        G.nodes[P+i]['bot']=1
        G.nodes[P+i]['leader']=0
        if len(botdict[l])>=round(botshare[numlead]*B) and numlead!=len(leaderlist)-1:
            numlead+=1
            while botshare[numlead]==0 and numlead!=len(leaderlist)-1:
                numlead+=1
        l=leaderlist[numlead]
        botdict[l].append(P+i)
        G.nodes[P+i]['position']=G.nodes[l]['position']
        G.nodes[P+i]['activity']=G.nodes[l]['activity']
        G.nodes[P+i]['pc']=G.nodes[l]['pc']
    for leader in leaderlist:
        for bot in botdict[leader]:
            G.add_edge(bot,leader)
            if l_a_b:
                G.add_edge(leader,bot)
            for bot2 in botdict[leader]:
                if bot!=bot2:
                    G.add_edge(bot,bot2)
    return G, botdict



def prune_feeds(G,tweets):
    global prng
    for V in G.nodes():
        feed= G.nodes[V]['feed']
        for p in feed:
            if (step+1)-tweets[p]['created']>lifespan:
                feed.remove(p)
        G.nodes[V]['feed']=feed
    return G



def node_actions(V,Ga):
    global similarity
    global lifespan
    global G
    global tweets
    global prng
    global current_feeds
    choice=prng.binomial(1,1/(Ga.nodes[V]['activity']-Ga.nodes[V]['clock']+1))+prng.binomial(1,Ga.nodes[V]['leader']*leader_boost)
    if choice>0:
        actions=[]
        current_feed=[]
        if step>0:
            if followings_reaction==1:
                if G.nodes[V]['leader']!=1 and len(G.nodes[V]['unreacted_followings'])>0:
                    new_followings_reaction(V)
            if (G.nodes[V]['bot']==1 and bots_retweet==1) or G.nodes[V]['bot']==0:
                current_feed=fill_current_feed(V,current_feed)
                current_feeds[V]=current_feed
                actions=process_feed(V,current_feed,actions)
        if step>0 and actions==[] and G.nodes[V]['bot']==1:
            actions=do_bot_things(V,actions)
        actions=write_tweet(V,actions)
        do_actions(V,actions)
    if choice>0:
        G.nodes[V]['clock']=0
    else:
        if G.nodes[V]['clock']<G.nodes[V]['activity']:
            G.nodes[V]['clock']=G.nodes[V]['clock']+1



def do_bot_things(V, actions):
    global leaderposts
    global prng
    global bots
    global G
    botdict=bots
    thisbotleader=0
    if necroposting==1:
        for leader in botdict.keys():
            if V in botdict[leader]:
                thisbotleader=leader
        lposts=leaderposts[thisbotleader]
        if len(lposts)>0:
            ch=round(prng.uniform(0,len(lposts)-1))
            chosen=leaderposts[thisbotleader][ch]
            actions.append(chosen)
            tweets[chosen]['retweets']+=1
            tweets[chosen]['retweeted_by'].append(V)
    if spam_following==1:
        nodes=set(list(G.nodes))
        followed=set([x[1] for x in G.out_edges(V)])
        candidates=list(nodes.difference(followed))
        for i in range(0,botsfollowings):
            choice=prng.binomial(1,subscription_probability/2)
            if choice>0 and G.nodes[V]['subscription_actions']<bot_subscriptions_limit:
                ch=round(prng.uniform(0,len(candidates)-1))
                chosen=candidates[ch]
                G.add_edge(V,chosen)
                changed_subscriptions.append(V)
                G.nodes[chosen]['unreacted_followings'].append(V)
                G.nodes[V]['subscription_actions']+=1

    return actions


def fill_current_feed(V,current_feed):
    global Ga
    global prng
    feed=Ga.nodes[V]['feed']
    if len(feed)>0:
        pc=Ga.nodes[V]['pc']
        out_edges=Ga.out_edges(V)
        dict_df={}
        for post in set(feed):
            if V not in tweets[post]['seen_by'] and post not in dict_df.keys():
                dict_df[post]={}
                dict_df[post]['indegree']=Ga.in_degree(tweets[post]['author'])
                dict_df[post]['total_retweets']=tweets[post]['retweets'] 
                dict_df[post]['subscribed_friends']=len(set([x[1] for x in out_edges]).intersection(set([x[1] for x in Ga.in_edges(tweets[post]['author'])])))
                dict_df[post]['number_in_feed']=feed.count(post)
        if len(dict_df.keys())>0:
            max_indegree=sorted([dict_df[post]['indegree'] for post in dict_df.keys()],reverse=True)[0]+1
            max_total_retweets=sorted([dict_df[post]['total_retweets'] for post in dict_df.keys()],reverse=True)[0]+1
            max_subscribed_friends=sorted([dict_df[post]['subscribed_friends'] for post in dict_df.keys()],reverse=True)[0]+1
           
            for post in dict_df.keys():
                dict_df[post]['score']=dict_df[post]['indegree']/max_indegree+dict_df[post]['total_retweets']/max_total_retweets+dict_df[post]['subscribed_friends']/max_subscribed_friends
                dict_df[post]['score']=dict_df[post]['score']*dict_df[post]['number_in_feed']
                
            dict_df={post:dict_df[post]['score'] for post in dict_df.keys()}
            sorted_feed=[k for k, v in sorted(dict_df.items(), key=lambda item: item[1],reverse=True)] 
            num=0
            while len(current_feed)<pc and num<len(sorted_feed):
                current_feed.append(sorted_feed[num])
                num+=1
    return current_feed



def process_feed(V,current_feed,actions):
    global tweets
    global G
    global Ga
    global subscription_probability
    global  changed_subscriptions
    global retweets_dict_active
    global retweets_dict
    global prng
    best_value=-10
    best=-1
    if len(current_feed)>0:
        for tweet in current_feed:
            if V not in tweets[tweet]['seen_by']:
                if tweets[tweet]['political']==1:
                    tw_pos=tweets[tweet]['position']
                    G.nodes[V]['influenced']+=tw_pos
                    G.nodes[V]['tweets_read']+=1
                    au_pos=Ga.nodes[V]['position']
                    tw_deg=Ga.in_degree(tweets[tweet]['author'])
                    tw_retw=tweets[tweet]['retweets_nonbots']/(len(tweets[tweet]['seen_by_nonbots'])+1)
                    if value_regime==0:
                        value=(1-sqrt((tw_pos-au_pos)*(tw_pos-au_pos)))*(log(tw_deg+1)+1)*(tw_retw+1)
                    elif value_regime==1:
                        value=(1-sqrt((tw_pos-au_pos)*(tw_pos-au_pos)))*(log(tw_deg+1)+1)
                    elif value_regime==2:
                        value=(1-sqrt((tw_pos-au_pos)*(tw_pos-au_pos)))*(tw_retw+1)
                    elif value_regime==3:
                        value=(log(tw_deg+1)+1)*(tw_retw+1)
                    elif value_regime==4:
                        value=1-sqrt((tw_pos-au_pos)*(tw_pos-au_pos))
                    elif value_regime==5:
                        value=log(tw_deg+1)+1
                    elif value_regime==6:
                        value=tw_retw+1
                    elif value_regime==7:
                        value=np.random.normal(0, 1, 1)
                    if sqrt((tw_pos-au_pos)*(tw_pos-au_pos))>tolerance:
                        unsubscription=prng.binomial(1,subscription_probability/2)
                        if unsubscription==1:
                            if tweets[tweet]['author'] in sorted([x[1] for x in G.out_edges(V)]):
                                G.remove_edge(V,tweets[tweet]['author'])
                                changed_subscriptions.append(V)
                            if V in retweets_dict_active:
                                if tweet in retweets_dict_active[V]:
                                    for retweeter in  retweets_dict_active[V][tweet]:
                                        if retweeter!=tweets[tweet]['author'] and retweeter in sorted([x[1] for x in G.out_edges(V)]):
                                            G.remove_edge(V,retweeter)
                                            if V not in changed_subscriptions:
                                                changed_subscriptions.append(V)
                
                else:
                    au_pos=Ga.nodes[V]['position']
                    tw_deg=Ga.in_degree(tweets[tweet]['author'])
                    tw_retw=tweets[tweet]['retweets']/(len(tweets[tweet]['seen_by'])+1)
                    if value_regime==0:
                        value=(1-sqrt((0-au_pos)*(0-au_pos)))*(log(tw_deg+1)+1)*(tw_retw+1)
                    elif value_regime==1:
                        value=(1-sqrt((0-au_pos)*(0-au_pos)))*(log(tw_deg+1)+1)
                    elif value_regime==2:
                        value=(1-sqrt((0-au_pos)*(0-au_pos)))*(tw_retw+1)
                    elif value_regime==3:
                        value=(log(tw_deg+1)+1)*(tw_retw+1)
                    elif value_regime==4:
                        value=1-sqrt((0-au_pos)*(0-au_pos))
                    elif value_regime==5:
                        value=log(tw_deg+1)+1
                    elif value_regime==6:
                        value=tw_retw+1
                    elif value_regime==7:
                        value=np.random.normal(0, 1, 1)
                if value>best_value:
                    best_value=value
                    best=tweet

        choice=prng.binomial(1,subscription_probability)
        if choice==1 and best!=-1:
            actions.append(best)
            tweets[best]['retweets']+=1 
            if Ga.nodes[V]['bot']==0:
                tweets[best]['retweets_nonbots']+=1
            tweets[best]['retweeted_by'].append(V)
            G.nodes[V]['influenced_retweets']+=tweets[best]['position']
            G.nodes[V]['retweets_made']+=1
            subscription=prng.binomial(1,subscription_probability)
            if subscription==1:
                if tweets[best]['author'] not in sorted([x[1] for x in G.out_edges(V)]):
                    G.add_edge(V,tweets[best]['author'])
                    changed_subscriptions.append(V)
                    G.nodes[tweets[best]['author']]['unreacted_followings'].append(V)
                    G.nodes[V]['subscription_actions']+=1
    return actions



def write_tweet(V,actions):
    global Ga
    global step
    global tweets
    global prng
    global nocats
    global leaderposts
    if Ga.nodes[V]['bot']==0 or Ga.nodes[V]['bot']*bots_write_tweets==1:
        choice=prng.binomial(1,subscription_probability)
        if choice>0:
            tweet={'seen_by':[V],'seen_by_nonbots':[V],'retweets':0,'retweets_nonbots':0,'created':step,'author':V,'retweeted_by':[]}
            political=1
            tweet['political']=1
            tweet['position']=Ga.nodes[V]['position']
            idt=len(tweets.keys())
            tweets[idt]=tweet
            actions.append(idt)
            if Ga.nodes[V]['leader']==1:
                leaderposts[V].append(idt)
    return actions



def do_actions(V,actions):
    global G
    global prng
    global retweets_dict
    followers=[x[0] for x in G.in_edges(V)]
    for t in actions:
        for f in followers:
            G.nodes[f]['feed'].append(t)
            if f not in retweets_dict:
                retweets_dict[f]={}
            if t not in retweets_dict[f]:
                retweets_dict[f][t]=[]
            retweets_dict[f][t].append(V)


def add_to_seen(current_feeds):
    global tweets
    for V in current_feeds.keys():
        for tweet in current_feeds[V]:
            if V not in tweets[tweet]['seen_by']:
                tweets[tweet]['seen_by'].append(V)
                if Ga.nodes[V]['bot']==0:
                    tweets[tweet]['seen_by_nonbots'].append(V)


def new_followings_reaction(V):
    global G
    global Ga
    followings_to_react=Ga.nodes[V]['unreacted_followings']
    for f in followings_to_react:
        subscription=prng.binomial(1,reciprocity_probability)
        if subscription==1 and f not in sorted([x[1] for x in G.out_edges(V)]):
            G.add_edge(V,f)
    G.nodes[V]['unreacted_followings']=[]


if not experiment_mode:
    experiment_parameters={'no':['experiment']}

for leaders_positions in leaders_positions_list:
    alpha,numleaders,max_activity,alpha2,lifespan,leader_boost,subscription_probability,totalP,b,tolerance,steps,botshareFirst,l_a_b,followings_reaction,botsfollowings,reciprocity_probability,bot_subscriptions_limit=get_defaults()
    

    globals()['b'] = param_b
    globals()['value_regime'] = param_regime
    globals()['bots_regime']=bots_regime
    for function in bots_regime_dict[bots_regime]:
        globals()[function]=bots_regime_dict[bots_regime][function]
    
    for bs in experiment_parameters['botshareFirst']:
        globals()['botshareFirst'] = bs
        seed = rnseed
        botshare=[botshareFirst,1-botshareFirst]
        first_leader=str(leaders_positions[0])
        second_leader=str(leaders_positions[1])
        print('b{}_reg{}_botreg{}_bs{}_fl_{}_sl{}_seed{}.json'.format(param_b, param_regime, bots_regime,  bs, first_leader, second_leader, seed))
        numleaders=len(leaders_positions)
        B=round(totalP*b) #number of bots
        P=totalP-B #number of real people
        prng = RandomState(rnseed)
        random.seed(rnseed)
        prng.seed(rnseed)
        leaderposts={}
        G,bots=generate_network(P)
        changed_subscriptions=[n for n in G.nodes()]
        tweets={}
        similarity_dict={}
        time=datetime.datetime.now()
        retweets_dict_active={}
        retweets_dict={}
        G_s=G.copy()
        Gs=G_s
        step=0

        if experiment_mode:
            filename=experiment_path+ 'G_init_b{}_reg{}_botreg{}_bs{}_fl_{}_sl{}_seed{}.json'.format(param_b, param_regime, bots_regime, bs, first_leader, second_leader, seed)
            with open(filename,'w') as js:
                json.dump(nx.node_link_data(G),js,cls=NumpyEncoder)
        for step in range(0,steps):
            Ga=G.copy()
            retweets_dict_active=retweets_dict
            retweets_dict={}
            if step>0:
                G=prune_feeds(G,tweets) 
            current_feeds={}
            for V in Ga.nodes():
                node_actions(V,Ga)
            add_to_seen(current_feeds)

            if experiment_mode:
                if step%experiment_graduality==0:
                    filename=experiment_path+ 'G_{}_b{}_reg{}_botreg{}_bs{}_fl_{}_sl{}_seed{}.json'.format(step, param_b, param_regime, bots_regime, bs, first_leader, second_leader, seed)
                    with open(filename,'w') as js:
                        json.dump(nx.node_link_data(G),js,cls=NumpyEncoder)
                    filename=experiment_path+ 'tweets_{}_b{}_reg{}_botreg{}_bs{}_fl_{}_sl{}_seed{}.json'.format(step, param_b, param_regime, bots_regime, bs, first_leader, second_leader, seed)
                    with open(filename,'w') as js:
                        json.dump(tweets,js,cls=NumpyEncoder)

        if experiment_mode:
            filename=experiment_path+ 'G_{}_b{}_reg{}_botreg{}_bs{}_fl_{}_sl{}_seed{}.json'.format(step, param_b, param_regime, bots_regime, bs, first_leader, second_leader, seed)
            with open(filename,'w') as js:
                json.dump(nx.node_link_data(G),js,cls=NumpyEncoder)
            filename=experiment_path+ 'tweets_{}_b{}_reg{}_botreg{}_bs{}_fl_{}_sl{}_seed{}.json'.format(step, param_b, param_regime, bots_regime, bs, first_leader, second_leader, seed)
            with open(filename,'w') as js:
                json.dump(tweets,js,cls=NumpyEncoder)


print('DONE')

