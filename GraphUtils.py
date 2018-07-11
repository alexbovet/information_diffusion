# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0


import os
from networkx import MultiDiGraph, write_graphml
try:
    import graph_tool.all as gt
except:
    pass
import numpy as np



os.system("taskset -p 0xff %d" % os.getpid())

    
def buildGraphSqlite(conn, graph_type, start_date, stop_date, 
                       hashtag_list_filter=None,
                       keyword_list_filter=None,
                       save_filename=None,
                       ht_group_supporters_include=None,
                       ht_group_supporters_exclude=None,
                       queries_selected=None,
                       additional_sql_select_statement=None,
                       graph_lib='graph_tool'):
    """ Returns graph for interaction types in `graph_type` from sqldatabase,
        using the graph library graph_lib.
        
        Notes
        -----
        tweets are selected such that `start_date` <= tweet timestamp < `stop_date`.
        
        if hashtag_list_filter is provided, only tweets containing hashtags in
        this list are used.

        if keyword_list_filter is provided, only tweets containing keywords in
        this list are used.
        
        `ht_group_supporters_include` and `ht_group_supporters_exclude` are used
        to select tweets with hashtags belonging to certain groups
        
        `additional_sql_select_statement` can be use to add a condition of the 
        tweet ids. Must start by "SELECT tweet_id FROM ... "
        
        user_ids are stored in a internal vertex_property_map named `user_id`.
        
        tweet_ids are stored in a internal edge_property_map names `tweet_id`.
        
        `graph_lib` can be 'graph_tool', 'networkx' or 'edge_list', where
        edge_list returns a numpy array of edges with (influencer_id, tweet_author_id, tweet_id)
        
    """
    
    c = conn.cursor()

    
    # transform the list of graph types to a list of table names
    graph_type_table_map = {'retweet': 'tweet_to_retweeted_uid',
                            'reply' : 'tweet_to_replied_uid',
                            'mention' : 'tweet_to_mentioned_uid',
                            'quote' : 'tweet_to_quoted_uid'}

    # table_name to influencer col_name                            
    table_to_col_map = {'tweet_to_retweeted_uid' : 'retweeted_uid',
                            'tweet_to_replied_uid': 'replied_uid',
                            'tweet_to_mentioned_uid' : 'mentioned_uid',
                            'tweet_to_quoted_uid' : 'quoted_uid'}  
                  
    table_names = []                            
    if isinstance(graph_type, str):
        if graph_type == 'all':
            table_names = list(graph_type_table_map.values())
        else:
            graph_type = [graph_type]
        
    if isinstance(graph_type, list):
        for g_type in graph_type:
            if g_type in graph_type_table_map.keys():
                table_names.append(graph_type_table_map[g_type])
            else:
                raise ValueError('Not implemented graph_type')
    
        
        
    table_queries = []
    values = []
    for table in table_names:
        
        sql_select = """SELECT tweet_id, {col_name}, author_uid
                     FROM {table} 
                     WHERE tweet_id IN 
                         (
                         SELECT tweet_id 
                         FROM tweet 
                         WHERE datetime_EST >= ? AND datetime_EST < ?
                         )""".format(table=table, col_name=table_to_col_map[table])
                         
        
        values.extend([start_date, stop_date])
    
        # add conditions on hashtags
        if hashtag_list_filter is not None:
            
            sql_select = '\n'.join([sql_select,
                    """AND tweet_id IN
                     (
                     SELECT tweet_id
                     FROM hashtag_tweet_user
                     WHERE hashtag IN ({seq})
                     )
                     """\
                     .format(seq = ','.join(['?']*len(hashtag_list_filter)))])
            
            for ht in hashtag_list_filter:
                values.append(ht)
    
        # add conditon on keyword
        if keyword_list_filter is not None:                             
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         SELECT tweet_id
                         FROM tweet_to_keyword
                         WHERE keyword IN ({seq})
                         )
                         """\
                         .format(seq = ','.join(['?']*len(keyword_list_filter)))])                      
                             
            for kw in keyword_list_filter:
                values.append(kw)
                
        if additional_sql_select_statement is not None:
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         """ + additional_sql_select_statement + """
                         )
                         """])
        #                 
        # intersect with given ht groups
        #
        if ht_group_supporters_include is not None:
            sql_included_groups = []
            for ht_gn in ht_group_supporters_include:
                sql_included_groups.append("""SELECT tweet_id 
                                        FROM hashtag_tweet_user
                                        WHERE ht_group == '{htgn}'""".format(htgn=ht_gn))
                
            sql_included_groups = '\nUNION ALL\n'.join(sql_included_groups)
                
            
            if ht_group_supporters_exclude is not None:
                sql_excluded_groups = []
                for ht_gn in ht_group_supporters_exclude:
                    sql_excluded_groups.append("""SELECT tweet_id 
                                    FROM hashtag_tweet_user
                                    WHERE ht_group == '{htgn}'""".format(htgn=ht_gn))
        
                sql_excluded_groups = '\nUNION ALL\n'.join(sql_excluded_groups)

            
                sql_select = '\n'.join([sql_select,
                            """AND tweet_id IN
                             (
                             SELECT * FROM (""" + sql_included_groups + """)
                             EXCEPT
                             SELECT * FROM (""" + sql_excluded_groups + """)
                             )
                             """])
                             
            else:
                sql_select = '\n'.join([sql_select,
                            """AND tweet_id IN
                             (
                             """ + sql_included_groups + """
                             )
                             """])
                         
        #
        # intersect with queries selected
        #
        if queries_selected is not None:                          
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         SELECT tweet_id
                         FROM tweet_to_query_id
                         WHERE query_id IN (
                                            SELECT id 
                                            FROM query 
                                            WHERE query IN ('{qulst}')
                                            )
                         )
                         """\
                         .format(qulst = "','".join(queries_selected))])                      
                             

            
        table_queries.append(sql_select)
        
    # take union of all the interaction type tables
    sql_query = '\nUNION \n'.join(table_queries)
    
                              
#    print(sql_query)                
    c.execute(sql_query, values)
                     
    if graph_lib == 'graph_tool':
        G = gt.Graph(directed=True)
        G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
        G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')
    
        edge_list = np.array([(infl_uid, auth_uid, tweet_id ) for tweet_id, 
                              infl_uid, auth_uid in c.fetchall()],
                              dtype=np.int64)
    
        G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id])
    
            
        if save_filename is not None:
            G.save(save_filename)
            
    elif graph_lib == 'networkx':
        G = MultiDiGraph(graph_type=', '.join(graph_type))
        
        
        G.add_edges_from([(infl_uid, auth_uid, {'tweet_id': tweet_id}) for tweet_id, 
                              infl_uid, auth_uid in c.fetchall()])
        
        if save_filename is not None:
            write_graphml(G, save_filename)
            
    elif graph_lib == 'edge_list':
        G = np.array([(infl_uid, auth_uid, tweet_id ) for tweet_id, 
                              infl_uid, auth_uid in c.fetchall()],
                              dtype=np.int64)
        
        
    return G
    
