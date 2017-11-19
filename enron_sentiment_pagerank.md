
TODO:
define the mechanics of the algorithm
As part of this research a tool was developed as a proof of concept for stakeholders to
visualise their organisation


<center>
# Are Managers More Miserable?: Finding Sentiment Leaders in a Social Network.

**Ishaan Varshney**
University of New South Wales
Sydney, NSW 2052, Australia
</center>
### Abstract
Discovering opinion leaders in a social network has often been of interest to researchers and corporations as they can work with such leaders for information dissapation. However, current work has primarily focussed on generic measurements (e.g. degree, closeness centrality) which does not take adavantage of context as well as the emotional tone of communication in the network. In this paper, by leveraging advancements in sentiment polarity of NLP systems, we propose a new algorithm: StressRank. StressRank aims to rank individuals for each polarity of sentiement of their incoming messages. The idea being that if you recieve an email from an indiviual who themselves have high poliarity in a particular direction, then an email from them would contain that same polarity. w This algorithm was attempted against the Enron email corpus. Given the strong assumptions this paper takes, it was found that Managers on average recieve more positive emails than their subordinates. As part of this research a visualisation tool was also developed for stakeholders to explore their social network.

### Introduction
The term 'social networks' is often considered synonymous with the prevelance of social media platforms however they have existed long before. The ability to develop and maintain large and complex social networks of H. sapiens of the genus Homo is often considered the reason for their success  from an evolutionary perspective. As more and more communication is mediated with the aid of technology, researchers have been able to develop accurate and useful models of the topology of social networks along with features such important nodes. These (which??) methods are often called vanity metrics[^smt] as they do not contextualise the data. E.g. one might just look at outdegree to determine the popularity however it might just be a bot whose messages people in the network mostly ignore. Thus, the need for other methods of analysis have been sought after. The most popular of such method is sentiement analysis[^pang].

    Social networks sites, in particular, are defined as web services where people can (1) construct a public or semipublic profile within a bounded system, (2) define a list of users with whom they establish a connection, and (3) view their list of connections and those made by the others within the system [13].

Sentiment analysis

How the two interact together.
Some previous examples of such work

Opinion leaders

Quanitfied self movement.
Capturing stress and depressions.
Wearables are capturing our physical attributes.
Can the same philosophy be applied to out emotional states.
Those under negativity? do they exhibit negitivity themselves.

In a social network setting, would those who are get an email from generally negative people more likely
to become negative.

### Preliminaries
define what is sentiment.
how we will be using sentiment.
define the graph.


### Sentiment Leader Algorithm


### Implementation

### Concluisons
This work can be used by organisations with a richer history to be develop insights in the organisation. Which individuals sentiments closely track company performance/price. Which employees have undergone a massive drop in incoming sentiment, these employees should


### Shortcomings
Looking are better threshold for polarity of sentiment classification.

### Further Work
As social netowrks become more location agnostic, some form of visualisation of company structure will become increasingly important. Another issue that arises with such remote work is being able to make proactive interventions regarding mental health of employees as it is significantly harder to guage emotional stress over written communication.

Perhaps some of these changes are to look at temporality.  In aggregate people sentiment changes and thus it averages out therefore
we do not see the any dicernable patterns in this paper.

Conduct on clean and perhaps better labeled dataset
perform with cleaner measures of sentiment.

Obtaining roles for more of the enron data would be good. A lot of conclusions are contingent on this assumption.

It would be interesting to see the change in outgoing behaviour from incoming behaviour or vice-versa. Would require a completely different algorithm

### Acknoledgements

The majority of this work was funded by BVN Architecture 255 Pitt St, Sydney NSW 2000.

<!--How to cite stuff-->
What is this? Yet *another* citation?[^fn3]

Some text in which I cite an author.[^fn1]

More text. Another citation.[^fn2]

[^smt]: http://www.socialmediatoday.com/social-business/2015-04-09/social-vanity-metrics-top-4-worst-offenders
[^pang]: Pang B., Lee L. Opinion mining and sentiment analysis. Found. Trends Inform. Retr. 2008;2(1–2):1–135.

[^fn1]: So Chris Krycho, "Not Exactly a Millennium," chriskrycho.com, July 22,
    2015, http://www.chriskrycho.com/2015/not-exactly-a-millennium.html
    (accessed July 25, 2015), ¶6.

[^fn2]: Contra Krycho, ¶15, who has everything *quite* wrong.

[^fn3]: Contra Krycho, ¶17.
