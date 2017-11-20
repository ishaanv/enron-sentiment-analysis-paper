<center>
# Are Managers More Miserable? -- Measuring stress in a social network with StressRank.

**Ishaan Varshney**  
University of New South Wales  
Sydney, NSW 2052, Australia  
</center>

### Abstract
Identifying individuals who are stressed or disatisfied in a network could be of great interest as it could reduce attrition and increase organisational cohesion, putting aside the obvious merits in increasing wellbeing of people.
However, current work has primarily focused on generic measurements (e.g. degree, closeness centrality) which does not take advantage of context and the emotional tone of communication in the network. In this paper, by leveraging advancements in sentiment polarity prediction of natural language processing systems, we propose a new algorithm: StressRank. StressRank aims to rank individuals in terms of importance with the flow of negative email. Importance of nodes in regards to positive email has been dubbed YesRank. This algorithm aims to capture that if one receives an email from an individual who themselves have high polarity in a particular direction, then an email from them would contain that same polarity. This algorithm was attempted against the Enron email corpus to test the hypothesis whether emails to executives' generally contain more positive emails or negative. Priminary analysis shows that executives' on average receive more positive emails than their subordinates. However, more research should be undertaken to incorporate both sentiments in one algorithm. As part of this research a visualisation tool was also developed for stakeholders to explore their social network ([link](https://ishaan.xyz/enron/index.html)).

### Introduction
##### Social networks
Online social networks can be defined as platforms where users can communication[^def]. Social networks can be distributed and private such as the emailing system at at a corporation or more public like sites such as Facebook and Twitter. The ability to develop and maintain large and complex social networks of H. sapiens of the genus Homo is often considered the reason for their success from an evolutionary perspective. As more and more communication is mediated with the aid of technology, researchers have been able to develop accurate and useful models of the topology of social networks along with features such important nodes. This field of study is called Social Network Analysis (SNA). The most commonly used SNA methods are ones that provide some information about the structure of the graph. For example, closeness centrality provides an indication of which nodes are most often on the shortest path between nodes. Another example is degree centrality, which provides the number of incoming or outgoing connections a node may have. However, these measures are often called vanity metrics[^smt] as they do not contextualise the data and may lead to spurious insights. In an email network, one might just look at out-degree to determine the importance of the node however it might just be an bot whose messages people in the network mostly ignore. Thus, the need for other methods of analysis of SAs have been sought after. The most popular of such methods is by leveraging breakthroughs in the field of sentiment analysis (SA)[^pang].

##### Sentiment Analysis
Sentiment analysis (SA) has been one of the most active research areas in natural language processing (NLP)[^lui]. This is especially true given the recent success of artificial neural networks in such tasks. The aim of sentiment analysis is to "define automatic tools able to extract subjective information in order to create structured and actionable knowledge."[^book]. Polarity classification is a branch of SA which is concerned with extracting positive or negative sentiment from text. Long short term memory (LSTM) networks have excelled[^OAIPaper] in the NLP domain and many such trained networks have been open sourced to the public such as the sentiment neuron by OpenAI[^sentneu].


##### Combining Sentiment Analysis and Social Network Analysis
Current research has shown that SA and SNA have a complementary relationship in increasing the explanatory and predictive value of each other. Some previous examples of such work include using Twitter follows and mentions with an SA model for better predicting attitudes about political and social events[^tan].


### Preliminaries
##### Definitions
In the scope of this paper, a social network is defined by the communication flows between individuals. Each flow has a source, a target and some message content $$c$$. To calculate the sentiment we have a model _f_ which takes the content of the message and output as quantitative rating with 0 depicting neural sentiment, positive values depicting positive sentiment and negative values for negative sentiment.

##### StressRank Algorithm
1. Generate a quantitative measure of sentiment polarity with **negative** values depicting negative sentiment
1. Filter the dataset such that only positive polarities are present.
1. Generate an square matrix $$S$$ with the dimensions being the number of nodes in the system.

    $$
     S_{ij} = \sum_tf(c_{ijt})
    $$ 
     
    Each entry in the matrix is cumulative sentiment over all messages $$t$$ sent from user $$i$$ to $$j$$.
1. If a user is 'dangling', that is they have only recieved emails and did not sent any. They distribute their positivity or negativity evenly among all other users.
2. Normalise the matrix to ensure it is row stocastic.
1. Run the PageRank algorithm with $$\alpha = \dfrac{7}{8}$$.

##### YesRank Algorithm
We follow the same steps as above except in step 1 we only analyise **positive** sentiment.

### Implementation
#### Enron corpus
TODO add a bit about which subset of the data was used since it was so huge and what assumptions does that imply.

The enron corpus is often used the canonical example for exploring new methods in SNA. Using the public MySQL database[^mysql], the email bodies were classified using OpenAI's sentiment neuron[^sentneu] which has set a new benchmark on the Stanford SST with minimal training[^OAIPaper].

##### StressRank Results
|Person|Title| StressRank|
| ---------------------------|:-----------------:| -----:|
|john.lavorato@enron.com     | CEO               |0.094792432846432004|
|louise.kitchen@enron.com    | President         |0.088307953845699255|
|chris.germany@enron.com     | Employee          |0.079346235238653501|
|jeff.dasovich@enron.com     | Employee          |0.058376755251766797|
|drew.fossum@enron.com       | Vice President    |0.04579269766449149|
|kay.mann@enron.com          | Employee          |0.043860806139328697|



##### YesRank Results

|Person|Title| YesRank|
| ---------------------------|:-----------------:| -----:|
|john.lavorato@enron.com   | CEO            | 0.13227112485640746|
|louise.kitchen@enron.com  | President      |0.095584053443868972|
|david.delainey@enron.com  | CEO            | 0.079763971159627606|
|jeff.dasovich@enron.com   | Employee       |0.074223618879691838|
|richard.shapiro@enron.com | Vice President |0.066082559377388772|
|chris.germany@enron.com   | Employee       |0.035021913277732013|

Preliminary findings how that typically executive management have higher YesRank than employees. The same seems to be true for StressRank. However, assumptions could completely be wrong and should be applied repeated with a more complete social network. 

### Conclusions
This work can be used by organisations with a richer history to be develop insights in the organisation. Which individuals sentiments closely track company performance/price. Which employees have undergone a massive drop in incoming sentiment, these employees.


### Further Work
Looking are better threshold for polarity of sentiment classification. Needs some sort of human classification.

As social networks become more location agnostic, some form of visualisation of company structure will become increasingly important. Another issue that arises with such remote work is being able to make proactive interventions regarding mental health of employees as it is significantly harder to gauge emotional stress over written communication.

The other side of the coin could look at developing a model to use a kind of a recommender system for finding good fits in the cliques of the social network.

Yet another avenue could just be using the tool that comes

Perhaps some of these changes are to look at temporality.  In aggregate people sentiment changes and thus it averages out therefore
we do not see the any dicernable patterns in this paper.

Conduct on clean and perhaps better labeled dataset
perform with cleaner measures of sentiment.

Obtaining roles for more of the enron data would be good. A lot of conclusions are contingent on this assumption.

It would be interesting to see the change in outgoing behaviour from incoming behaviour or vice-versa. Would require a completely different algorithm.

Ethics of doing this kind of research. All of this should be done with transparency.

### Appendix

##### Python code to compute StressRank
```python
import numpy as np
import pandas as pd
from numpy import linalg as LA
from tqdm import tqdm


def sending_sentiment_adj_mat(mat, df, source_field='from', target_field='to'):
    for index, row in tqdm(df.iterrows()):
        source_index = nodes.index(getattr(row, source_field))
        target_index = nodes.index(getattr(row, target_field))
        roles[source_index] = getattr(row, 'from_title')
        roles[target_index] = getattr(row, 'to_title')
        mat[source_index][target_index] += row.sentiment


def normalise_matrix(G):
    sum_rows = G.sum(axis=1)
    for index, row_sum in enumerate(sum_rows):
        for col in range(G.shape[0]):
            G[index][col] = G[index][col] / row_sum
    return G


def add_dangling(G):
    sum_rows = G.sum(axis=1)
    for index, row_sum in enumerate(sum_rows):
        if row_sum == 0:
            for col in range(G.shape[0]):
                G[index][col] = 1 / G.shape[0]
    return G


enron_links = pd.read_csv('data/enron_links_with_sentiment_roles.csv')
enron_links = enron_links[enron_links.sentiment > 0]  # get positive sentiment
nodes = pd.unique(enron_links[['from', 'to']].values.ravel('K')).tolist()
n = len(nodes)
roles = [None] * n
mat = np.zeros([n,n])
sending_sentiment_adj_mat(mat, enron_links)
mat = add_dangling(mat)
mat = normalise_matrix(mat)
mat2 = np.array([[1 / mat.shape[0]] * mat.shape[0]] * mat.shape[0])
alpha = 7 / 8
G = alpha * mat + (1 - alpha) * mat2
w, v = LA.eig(G.T)
rank_vec = v[:, np.argmax(w)]
rank_vec = rank_vec / rank_vec.sum()
scores = [(nodes[i],roles[i], v) for i, v in enumerate(rank_vec)]
for i in sorted(scores, key=lambda x: x[2], reverse=True):
    print(i)
    
```







### References

[^smt]: http://www.socialmediatoday.com/social-business/2015-04-09/social-vanity-metrics-top-4-worst-offenders
[^pang]: Pang B., Lee L. Opinion mining and sentiment analysis. Found. Trends Inform. Retr. 2008;2(1–2):1–135.
[^tan]: Tan C., Lee L., Tang J., Jiang L., Zhou M., Li P. User-level sentiment analysis incorporating social networks. In: Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM; 2011:1397–1405.
[^mysql]: http://www.ahschulz.de/enron-email-data/
[^lui]: Liu B. Sentiment Analysis and Opinion Mining. San Rafael, CA: Morgan & Claypool; 2012.
[^book]: Pozzi, Federico Alberto, et al. Sentiment Analysis in Social Networks, Elsevier Science, 2016. ProQuest Ebook Central, https://ebookcentral-proquest-com.wwwproxy1.library.unsw.edu.au/lib/unsw/detail.action?docID=4713944.
[^sentneu]: https://github.com/openai/generating-reviews-discovering-sentiment
[^OAIPaper]: https://arxiv.org/pdf/1704.01444.pdf
[^def]: Ellison N.B. Social network sites: definition, history, and scholarship. J. Comput. Mediat. Commun. 2007;13(1):210–230.  