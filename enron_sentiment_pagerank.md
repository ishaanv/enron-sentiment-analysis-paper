<center>
# Are Managers More Miserable?: Finding Sentiment Leaders in a Social Network.

**Ishaan Varshney**
University of New South Wales
Sydney, NSW 2052, Australia
</center>
### Abstract
Discovering opinion leaders in a social network has often been of interest to researchers and corporations as they can work with such leaders for information dissapation. However, current work has primarily focussed on generic measurements (e.g. degree, closeness centrality) which does not take adavantage of context as well as the emotional tone of communication in the network. In this paper, by leveraging advancements in sentiment polarity of NLP systems, we propose a new algorithm: StressRank. StressRank aims to rank individuals for each polarity of sentiement of their incoming messages. The idea being that if you recieve an email from an indiviual who themselves have high poliarity in a particular direction, then an email from them would contain that same polarity. w This algorithm was attempted against the Enron email corpus. Given the strong assumptions this paper takes, it was found that Managers on average recieve more positive emails than their subordinates. As part of this research a visualisation tool was also developed for stakeholders to explore their social network.

### Introduction
##### Social networks
The term 'social networks' is often considered synonymous with the prevelance of social media platforms however they have existed long before. The ability to develop and maintain large and complex social networks of H. sapiens of the genus Homo is often considered the reason for their success  from an evolutionary perspective. As more and more communication is mediated with the aid of technology, researchers have been able to develop accurate and useful models of the topology of social networks along with features such important nodes. This field of study is called Social Network Analysis. The most common SNA methods are ones that provide some information about the structure of the graph. For example closeness centrality provides an indication of which nodes are most often on the shortest path between nodes. Degree centrality provides the number of incoming or outgoing connections a node may have. However, these measures are often called vanity metrics[^smt] as they do not contextualise the data and may lead to spurious insights. For example, one might just look at outdegree to determine the popularity however it might just be a bot whose messages people in the network mostly ignore. Thus, the need for other methods of analysis have been sought after. The most popular of such method is sentiement analysis[^pang].

    Social networks sites, in particular, are defined as web services where people can (1) construct a public or semipublic profile within a bounded system, (2) define a list of users with whom they establish a connection, and (3) view their list of connections and those made by the others within the system [13].

##### Sentiment Analysis
// TODO just some history about sentiment analysis



##### Combining Sentiment Analysis and Social Network Analysis
Current research has shown that SA and SNA
have a complementary relationship in increasing the expanitory and predictive value of each other. Some previous examples of such work include using Twiter follows and mentions with an SA model for better prediciting attitudes about political and social events[^tan].



### Preliminaries
##### Definitions
// sentiment: assumed negative information means negative email
// source
// target.

##### Method Overview
1. Given the communication content, source and target of the communition. Generate a quantitative measure of sentiment polarity with positive values depicting positive sentiment

### Sentiment Leader Algorithm
pagerank definition but instead links are normalised sentiments, such that probability to send a negative email is just the number of negative emails. with a random surfer.

### Implementation
##### Enron corpus
Using the public MySQL database[^mysql], the email bodies were classified using

### Concluisons
This work can be used by organisations with a richer history to be develop insights in the organisation. Which individuals sentiments closely track company performance/price. Which employees have undergone a massive drop in incoming sentiment, these employees should


### Shortcomings
Looking are better threshold for polarity of sentiment classification.

### Further Work
As social netowrks become more location agnostic, some form of visualisation of company structure will become increasingly important. Another issue that arises with such remote work is being able to make proactive interventions regarding mental health of employees as it is significantly harder to guage emotional stress over written communication.

The other side of the coin could look at develpoing a model to use a kind of a recommender system for finding good fits in the cliques of the social network.

Yet another avenue could just be using the tool that comes

Perhaps some of these changes are to look at temporality.  In aggregate people sentiment changes and thus it averages out therefore
we do not see the any dicernable patterns in this paper.

Conduct on clean and perhaps better labeled dataset
perform with cleaner measures of sentiment.

Obtaining roles for more of the enron data would be good. A lot of conclusions are contingent on this assumption.

It would be interesting to see the change in outgoing behaviour from incoming behaviour or vice-versa. Would require a completely different algorithm

### Acknoledgements

The majority of this work was funded by BVN Architecture 255 Pitt St, Sydney NSW 2000.


[^smt]: http://www.socialmediatoday.com/social-business/2015-04-09/social-vanity-metrics-top-4-worst-offenders
[^pang]: Pang B., Lee L. Opinion mining and sentiment analysis. Found. Trends Inform. Retr. 2008;2(1–2):1–135.
[^tan]: Tan C., Lee L., Tang J., Jiang L., Zhou M., Li P. User-level sentiment analysis incorporating social networks. In: Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM; 2011:1397–1405.
[^mysql]: http://www.ahschulz.de/enron-email-data/
