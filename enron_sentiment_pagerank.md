TODO put results in table.

<center>
# Are Managers More Miserable? -- Finding Sentiment Leaders in a Social Network.

**Ishaan Varshney**  
University of New South Wales  
Sydney, NSW 2052, Australia  
</center>
### Abstract
Identifying opinion leaders in a social networks has often been of interest to researchers and industry as they can work with such leaders for information dissipation marketing, extremist ideology etc. However, current work has primarily focused on generic measurements (e.g. degree, closeness centrality) which does not take advantage of context as well as the emotional tone of communication in the network. In this paper, by leveraging advancements in sentiment polarity of NLP systems, we propose a new algorithm: StressRank. StressRank aims to rank individuals for each polarity of sentiment of their incoming messages. The idea being that if you receive an email from an individual who themselves have high polarity in a particular direction, then an email from them would contain that same polarity. w This algorithm was attempted against the Enron email corpus. Given the strong assumptions this paper takes, it was found that Managers on average receive more positive emails than their subordinates. As part of this research a visualisation tool was also developed for stakeholders to explore their social network.

### Introduction
##### Social networks
The term _social networks_ is often considered synonymous with the prevalence of social media platforms however they have existed long before. The ability to develop and maintain large and complex social networks of H. sapiens of the genus Homo is often considered the reason for their success  from an evolutionary perspective. As more and more communication is mediated with the aid of technology, researchers have been able to develop accurate and useful models of the topology of social networks along with features such important nodes. This field of study is called Social Network Analysis. The most common SNA methods are ones that provide some information about the structure of the graph. For example closeness centrality provides an indication of which nodes are most often on the shortest path between nodes. Degree centrality provides the number of incoming or outgoing connections a node may have. However, these measures are often called vanity metrics[^smt] as they do not contextualise the data and may lead to spurious insights. For example, one might just look at out-degree to determine the popularity however it might just be a bot whose messages people in the network mostly ignore. Thus, the need for other methods of analysis have been sought after. The most popular of such method is sentiment analysis[^pang].

    Social networks sites, in particular, are defined as web services where people can (1) construct a public or semipublic profile within a bounded system, (2) define a list of users with whom they establish a connection, and (3) view their list of connections and those made by the others within the system [13].

##### Sentiment Analysis
// TODO just some history about sentiment analysis
Sentiment analysis (SA) has been one of the most active research areas in natural language processing (NLP)[^lui]. This is especially true given the rise in popularity and success of artificial neural networks in such tasks. The aim of sentiment analysis is to "define automatic tools able to extract subjective information in order to create structured and actionable knowledge."[^book]. Polarity classification is a branch of SA which is concerned with extracting positive or negative sentiment from text. Long short term memory (LSTM) networks have excelled[^OAIPaper] in the NLP domain and many such trained networks have been open sourced to the public such as the sentiment neuron by OpenAI[^sentneu].


##### Combining Sentiment Analysis and Social Network Analysis
Current research has shown that SA and SNA
have a complementary relationship in increasing the explanatory and predictive value of each other. Some previous examples of such work include using Twitter follows and mentions with an SA model for better predicting attitudes about political and social events[^tan].


### Preliminaries
##### Definitions
In the scope of this paper, social network is defined by the communication flows between individuals. Each flow has a source, a target and the content of the message. To calculate the sentiment we have a model _M_ which takes the content of the message and output as quantitative rating with 0 depicting neural sentiment, positive values depicting positive sentiment and negative values for negative sentiment.

##### Method Overview
1. Given the communication content, source and target of the communication. Generate a quantitative measure of sentiment polarity with positive values depicting positive sentiment
1. Filter the dataset such that only positive polarities are present.
1. Generate an square matrix $$S$$ with the dimensions being the number of nodes in the system.

    $$
     S_{ij} = \sum_tf(c_{ijt})
    $$ 
     
    Each entry in the matrix is cumulative sentiment over all messages sent from user i to j.
1. Run the PageRank algorithm

### Sentiment Leader Algorithm
Generally talk about the modification done to the data to make it fit the pagerank paradigm and what does it signify

### Implementation
#### Enron corpus
Using the public MySQL database[^mysql], the email bodies were classified using

##### Results


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
[^lui]: Liu B. Sentiment Analysis and Opinion Mining. San Rafael, CA: Morgan & Claypool; 2012.
[^book]: Pozzi, Federico Alberto, et al. Sentiment Analysis in Social Networks, Elsevier Science, 2016. ProQuest Ebook Central, https://ebookcentral-proquest-com.wwwproxy1.library.unsw.edu.au/lib/unsw/detail.action?docID=4713944.

[^sentneu]: https://github.com/openai/generating-reviews-discovering-sentiment

[^OAIPaper]: https://arxiv.org/pdf/1704.01444.pdf
