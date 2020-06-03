<!-- wp:separator {"className":"is-style-wide"} -->
<hr class="wp-block-separator is-style-wide"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3> </h3>
<!-- /wp:heading -->

<!-- wp:quote {"align":"center"} -->
<blockquote class="wp-block-quote has-text-align-center"><p>Use my <a href="https://colab.research.google.com/drive/1kET5ylMpJADqbFaIstpLfK8zLfxEXeOh?usp=sharing"><strong><span style="text-decoration:underline;">Google Colab Notebook</span></strong></a> for interactive learning!</p></blockquote>
<!-- /wp:quote -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/a3010-1vhkfdxxvtqrtscksbkltvg.jpeg" alt=""/><figcaption>How are&nbsp;you?</figcaption></figure>
<!-- /wp:image -->

<!-- wp:heading {"level":4} -->
<h4><strong>What is Sentimental Analysis?</strong></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Every day, whatever we do always linked with our Emotions. In each condition, we all have some specific attitudes towards the event. Way of representing such emotions may be different, but our reaction always tells some story.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Applying the analytical methodology to determine an individual’s opinion and categorize it to specific emotion (positive, negative, neutral, happy, sad, angry, humble etc…) So, in summary, “Sentiment analysis is the interpretation and classification of emotions.”</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Applications:&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>Natural language processing (NLP), text analysis, computational linguistics Analysis.</li><li>For security reasons such as biometrics to systematically identify, extract, quantify, and study affective states and subjective information.</li><li>Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine and many more…</li></ul>
<!-- /wp:list -->

<!-- wp:heading {"level":4} -->
<h4>Now, let’s come to the point that how recent development of <a href="https://en.wikipedia.org/wiki/Machine_learning" rel="noreferrer noopener" target="_blank">Machine-learning</a> can help us to “dig gold out of the trash”(row dataset).</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>My main focus of this topic is to discuss neural networks in depth. but still, let’s start with methods commonly used for Sentimental Analysis.</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li><a href="https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/" rel="noreferrer noopener" target="_blank">Naive Bayes</a></li><li><a href="https://hackernoon.com/supervised-machine-learning-linear-regression-in-python-541a5d8141ce" rel="noreferrer noopener" target="_blank">Linear Regression</a></li><li><a href="https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/" rel="noreferrer noopener" target="_blank">Support Vector Machine</a></li><li><a href="https://www.youtube.com/watch?v=aircAruvnKk" rel="noreferrer noopener" target="_blank">Neural Network</a> (MLP, CNN, RNN)</li></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3>Why Neural&nbsp;Network?&nbsp;</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Normally, we see large use of the NLTK library to preprocessing text mining data sets and then direct application of classifier to get output. But, nowadays the scenario is changed…</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In the world of big data, everyone has 5 V problems when we practically deal with data. One of them is the Velocity of DATA. means <strong>“The Data generation speed is higher than our Data analytic capability, in terms of hardware, human resource, analytic methodologies, and algorithmic limitation.”</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>To extend our limitations, we use the neural network which can able to find our complex combination of text data using the RNN/LSTM model. Even if we have data in terms of face detection (as first product review) or videography still we can find the best approach using CNN.</p>
<!-- /wp:paragraph -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>Well, my main motto of this article is to give depth understanding of fundamentals, preprocessing (for neural network models), Network architecture explaination and applying data Noise reduction concepts.&nbsp;</p></blockquote>
<!-- /wp:quote -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3>Commence:&nbsp;</h3>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Here, I will use the review and label data in the form of a text file (.txt). You can download the dataset and my google Colab Notebook for an interactive tutorial. then just run following code:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/e5ba4-1be1f_9xtq7dfpinewitsew.png" alt=""/><figcaption>Load Data</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/fb4a7-1gsgyvywwgf__o3sqhrsuna.png" alt=""/><figcaption>Preview Nature of&nbsp;Data&nbsp;</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>I represent each and every step in detail so, please forgive me for lengthy reading. But, I am sure that you will understand the whole application in detail.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>In each stage,<strong> first I will explain “appllied fundamental/hypothesis”, then relevant “codes with code logic”, and “resultatnt outputs” for unit test models and then apply to Main model. All sequences will maintain until end and seperated by step&nbsp;: X.x (where X=Stage, x= substeps) </strong>By the way, I will partitioning this in following 3 stages: [1] Feature Extraction &amp; I.P.-O.P. creation. [2] Model Architecture(NN). [3] Noise Reduction.</p></blockquote>
<!-- /wp:quote -->

<!-- wp:heading {"level":4} -->
<h4>Stage_1&nbsp;: Feature Extraction &amp; IP/OP creation&nbsp;:</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Features are an inherent property of an object which tell us how they differ from others as an Example:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol><li>“I am Happy”</li><li>“I am sad”</li></ol>
<!-- /wp:list -->

<!-- wp:list -->
<ul><li>In, both Happy and Sad are two different words that tell us the sentiment of a writer. So, for any text, their words are the ultimate representatives.&nbsp;</li><li>That means if we separate words from the text and generate total_counts that can create one vector of words as input.</li><li>In python&nbsp;<a href="https://www.w3schools.com/python/ref_string_split.asp" rel="noreferrer noopener" target="_blank">.split(“ “)</a> function will separate our words as they are separated in a text by blanks.</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p><strong>Step: 1.0</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/cb123-188pfxaixloflszpgavhc9a.png" alt=""/><figcaption>Generate Counters and Label Separators</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>I use Counters to count positive, negative, and total_words within their relevant labels. This will separate relevant words to identify the nature of the specific word represents in distinctive labels. And as you see below we get relevant words with their counts. Its collection of tuples in the form of (word, counts)</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/e0c4e-1jl7l5jopk5lgugspaymwpa.png" alt=""/><figcaption>.2</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Step: 1.1</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>But, from the above result we should not find meaningful information that can lead us to some conclusion, the words “the”, “and”, “a”, “of”, “to” are used a lot more time but do not relate to the sentiment of a writer so it's better to find ‘pos_neg_ratio” than this to reduce the word overlapping in each counter.</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>It will reduce computational and time complexity.</li><li>The positive-to-negative ratio for a given word can be calculated with <code>positive_counts[word] / float(negative_counts[word]+1)</code>. Notice the <code>+1</code> in the denominator – ensures, we don't divide by zero in case if words are only seen in positive reviews.</li><li>we will only consider words which are more than 100 times in total, its one kind of hypothesis which helps us to reduce words of product, brand name, or type of products.&nbsp;</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/609af-1umzclho_yfkv2qti899zkq.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/71440-1g2hofo4sh40nxw3zl7yk1a.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Result Interpretation:</strong> Here, our ration is positive to negative means if a word is in the positive label then, it has ration value &gt;1 and for negative terms value became &lt;1 (near to zero). Common words are around 1. You can find that in code result.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step: 1.2</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Now. our problem is the range of our pos_neg_ratio which is from +14 to 0.008. This will give us a large data distribution range and will make analysis hard in graphical representations. And, Larger digits of data will make the Neural networks hard to train and use more computation power. So here if we can transfer it to a small range with math then it's better for our model.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://danielmiessler.com/study/big-o-notation/" rel="noreferrer noopener" target="_blank"><strong><em>log(n)</em></strong></a>is one of them because log(2¹⁰⁰⁰⁰⁰⁰) ~= 20.00 (19.9315685693) so it will be better distribution band for us. <a href="https://stackoverflow.com/questions/2307283/what-does-olog-n-mean-exactly" rel="noreferrer noopener" target="_blank"><em>[click here for great Answers]</em></a><em>.</em></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/2048c-18cmbzdhtunihjvbdn92olw.png" alt=""/><figcaption>Observation: Matthau&nbsp;: current value = 2.80 reduced from 16.55 (previous value)</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/5f770-19u1hvo6iokzwfwzcdmjgug.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p><strong>Great!</strong>, now our<em> range </em>is from (+2.80 to -3.32) reduced from (16.55 to 0.02) and another advantage is <strong>distribution of data</strong> with reference to Y-axis cut X-axis at zero will give us <strong>levarage of better distribution, classification and visualization modeling.</strong></p></blockquote>
<!-- /wp:quote -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":4} -->
<h4>INPUT / OUTPUT DATA CREATION:</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>For, this we must understand what is input and output in terms of Neural_Network?</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/7b65e-1n3va7hsvtg6otf5founblg.png" alt=""/><figcaption>Simple FeedForward Neural-Network</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>It is, understandable that our words are input as we identified them as feature having power of deciding Writer’s Sentiment and Our output must be either Positive/Negative review. Means we must need softmax activation function to decide output in term of probability. If the network considers word probability (&gt;0.5) represent Positive Label and (&lt;0.5) as Negative Label. (I will discuss this in detail during the Network Architecture stage. So, Let’s decide on Input_layer (layer_0).&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step&nbsp;: 1.3 (Input generation)&nbsp;:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here, our measures are to create an input layer (layer_0) and use it to update layer_1(hidden_layer).&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/3cbb8-1ydj-1l-el8dydt2oxmdvta.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Interpretation:</strong> First we first create one set of total words and generate indexing to update information in layer_0. It's obvious that layer_0 should be and input length of our set_size (1,74074).&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step: 1.4 (Layer update):</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/3a7fd-1bjbcmzryrgx3yuidsxwfza.png" alt=""/><figcaption>Array[* * *] represents word repetition so the First word is repeated 18&nbsp;times.&nbsp;</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>First, I use <code>layer_0 *= 0</code> to reset the layer and then update it with for loop. <code>layer_0[0][word2Index[word]] +=1</code> counts each word of review and increment it for each repetition of the word. So, we get a real weightage of a word in each label. As an example, in a specific review “if customers use word satisfied or happy” 3 times then it increases its weight to output and will give a better result.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step&nbsp;: 1.4 (Output)&nbsp;:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Neural Network uses mathematical models and gives output in Numeric form. so, it's better to convert our output labels as <strong>1 and 0 for “POSITIVE” and “NEGATIVE”.&nbsp;</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/68905-1l7e1cygfi2wuitdiekva1g.png" alt=""/><figcaption>Ouput [Label Conversion ]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>So, here we completed our first milestone (Feature extraction and preprocessing stage) of Given Dataset. Our Next stage is Neural Network building.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong>Ok</strong>, so until now we discuss the idea behind the essence of sentimental analysis, reasons after using Neural_Network, text data preprocessing steps. Now, let’s move to Neural_Network Architecture.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4>The sequence of&nbsp;phases:</h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><strong><em>Generate class </em></strong><code>class Sentimental_Network:</code><strong><em>and Encapsulate Neural Network architecture</em></strong> with the following methods…</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>Define Data Preprocessing Method phase<code>def pre_processing_data()</code></li><li>Define Network architecture phase. <code>def network_architecture()</code></li><li>Training Model phase<code>def train()</code></li><li>Testing method phase<code>def test()</code></li><li>Run command <code>def run()</code></li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p><strong>Step&nbsp;: 2.0 (define class)</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/2cd9a-15ilxda49nuurc0einssi1w.png" alt=""/><figcaption><strong>Define class and called supportive methods</strong> [Continue…]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Python is Object Oriented Programming language, and here its class represents all four basic elements. You should check this <a href="https://www.programiz.com/python-programming/object-oriented-programming" rel="noreferrer noopener" target="_blank"><strong><em>Python OOP</em></strong></a> for more detail. We will use them in our code.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here, Sentimental_Network is defined class will contain the whole implementation of the network. “<strong>__init__” </strong>is a reserved method of python to implement other methods inside it. It’s known as a <strong>constructor</strong>. Here _ or __ is called <em>encapsulation</em>. So, after defining this we can not modify class further from outside of the network. “<strong><em>self”</em></strong> is also reserved for representing <strong><em>instance</em></strong>. well, self.pre_process_data() and self.network_architecture() are supportive method will discuss later below.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step&nbsp;: 2.1 (pre_processing phase)</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/1c386-1gmrx9mjleqwjitorf2vj6q.png" alt=""/><figcaption><strong>Define pre_processing_data() </strong>[Continue…]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>This method is already called in __init__, and as we already discussed it in [Part-1: Feature Extraction]. Still for a summary, first we generate one set and add populate set with words used in all reviews. The same process for labels and create a set of all POSITIVE &amp; NEGATIVE labels in the same sequence. In the second step we convert that set to a collection of tuples (word, index) to deal with input and output dataset after prediction.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step&nbsp;: 2.2 (Network_architecture phase)</strong></p>
<!-- /wp:paragraph -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>Again, I will consider every thing in detail so it is lengthy, sorry for that!</p></blockquote>
<!-- /wp:quote -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/0787c-1iysweivgedvbbdv7t541hq.png" alt=""/><figcaption><strong>Network Architecture</strong> [Continue…]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:list -->
<ul><li>With method <code>def network_architecture(self,#input,#output,lr)</code> we create required <strong>Input parameters</strong> to a neural network. here, as I defined we have no. of input nodes and output nodes. so, first of all, these input nodes generated with input_nodes. We have 74,074 words, so it will generate long tensor of size [1,74,074] for the input layer. other layers will be added during the training phase.</li><li><strong>Weight_initialization</strong> is one of the important parameters, how efficient you initialize your weight defines your network performance.&nbsp;</li></ul>
<!-- /wp:list -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/506c4-16c10amdltplk7ow9paq7fq.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:list -->
<ul><li>If you use this theorem to <code>hidden_nodes**-0.5</code>then it will give you a better result than <code>output_nodes**-0.5.</code> Because it will work with more neurons of hidden_layer, it helps to train network faster with 0.01 learning_rate than 0.001 (with output_nodes). [please, check it if you want].</li><li>We will talk <strong>learning_rate</strong> letter during the execution phase.</li><li>But, in <strong>Output,</strong> we just need to classify that “Is this review tagged with positive or negative? “ For that sigmoid function is used.&nbsp;</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Sigmoid function:</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/f286b-0g-h74fcpwzklftfi.png" alt=""/><figcaption>Sigmoid Activation function (probability — (0.0,1.0))</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/aed1f-1mfr89ki4lurojuiijfw9wg.png" alt=""/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Here, the sigmoid function is simplified form of “<strong><em>(single element) power of e give the numeric value of the single element and it is divided by power(sum(elements)) of e” </em></strong>will return ratio and its value will be between (0 and 1). If the probability is above 0.5 then it referred to POSITIVE(1) otherwise NEGATIVE(0). [<a href="https://en.wikipedia.org/wiki/Sigmoid_function" rel="noreferrer noopener" target="_blank"><em>Sigmoid function in detail</em></a>]</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Step: 2.3 (define Training phase)</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/96911-1wbsysxpwddvexqcvz6aooq.png" alt=""/><figcaption><strong>Neural_Net Training</strong> [Continue…]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>For training, input and output are mandatory parameters. Here, we must ensure the number of inputs and number of outputs that they must match to each other, otherwise even after training, we should not match them properly and getting errors (“out of bounding [index]” ).&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Every neural network works with individual data. In simple classification modeling, NeuralNet actually tries to plot our data in a 2D or 3D graph and classify(divide) them with one line (linear/nonlinear) or plane. For that, it passes through different layers and each time reduced its layer size by making different patterns. During prediction, these patterns are associated with some specific output and the degree of closeness of these patterns to actual output gives us probability and highest probable scenario considered as output. [please read it twice].&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/4541a-1ko0gj8vbeudhnomggdza7a.jpeg" alt=""/><figcaption><strong>&nbsp;Forward [green] &amp; BackPropogation [red]</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>As shown above,&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>&nbsp;The first input(X) and W_0_1 dot-matrix multiplication give intermediate output as layer_1 and in case of an activation function, it will work as filter as per its formula. then this flow works the same way with new weights of layers. And finally give output (pos/neg) as per value x ( for, x &lt; or &gt; 0.5)</li><li>When you get actual output then Forward networks complete itself but still when we match our network output with actual targets(real labels) then it gives the error. In starting, errors are quite bigger than our expectations. But, here the main essence of backpropagation works.</li><li>Backpropagation take total errors and reflects it to layers in reverse order(means distribute total errors in layers and each layer’s individual neurons (according to their weight distribution). In this way now each neuron has a chance to adjust their weights to maximize output probability.&nbsp;</li><li>Let’s understand it in terms of patterns. In that concern, each neurons’ weight updates their self in a way that a more important pattern’s neuron’s weight will become bigger than others. In such a way the total effect of such important neurons effectively high and give better output.&nbsp;</li></ul>
<!-- /wp:list -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>This way our neural network improve its performance by adjusting weights and biases(here,bias=0), called tuning parameters accordingly.&nbsp;</p></blockquote>
<!-- /wp:quote -->

<!-- wp:paragraph -->
<p><strong>Step: 2.4 (define the testing phase)</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/03687-1twoe3k1uyoul_7e2zyjwwq.png" alt=""/><figcaption><strong>Model Testing</strong> [Continue…]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The testing phase only contains forward propagation and by defining run() method we get perfect output. In testing, we just need to run Forward pass (same as training) and compare our achieved output with a real one. Here, besides getting error we can get true or false for our result.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":4} -->
<h4>Step: 2.5 Execution Phase:&nbsp;</h4>
<!-- /wp:heading -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/66fc7-1xmn1hefvbbfp3wvufwz9bq.png" alt=""/><figcaption><strong>Model Execution</strong> [Demo]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Here, first I define model_0 with proper parameters, I take all data except last 1000 for network defining and training mode. Because testing data always new (unknown) for the network. Here, during execution learning_rate = 0.1 and as I declared before, it is one of hyperparameter for NeuralNet for this we should understand the learning rate in depth.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>Learning_rate</strong></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>In our code, we use the learning rate to update weights. so, according to mathematical modeling, we use it during weight update in backpropagation. It defines how fast our weight adjusts themselves. Let’s understand by example</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><em>Example:</em></strong> If our network has previous weight <strong>(W_old= 0.5) </strong>and now backpropagation advice it to reset by some part of error (let’s take 0.1) that means new weight <strong>“W_new” will becomes [0.5 – 0.1 = 0.4]</strong> but our formula suggest to multiply it by learning_rate then as per formula same update will work as <strong>W_new = [0.5 – (learning_rate= 0.1) * 0.1 = 0.49]</strong>. But, what if our learning rate is 0.01? obviously it gives value <strong>W_new = [0.5 — (learning_rate= 0.01) * 0.1 = 0.499]. </strong>In this way, learning_rate decide the real output.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>If you try to think <em>mathematically</em> then its rate of network’s gradient descent which decides the slope of our network’s loss_function that how steep our curve will be. and how fast we reach the lowest point of loss curve(smallest error,&nbsp;… means the likelihood of real value maximization) as below.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Generally, ML practitioner uses its range from 0.001 to 0.9 but my personal choice is to first check 0.01, and try 0.1 and 0.001 and from the graph slope I try to choose learning_rate further. Because when you are going to use (0.0001,0.001,0.01,0.1,1), you are actually working in a specific manner called exponential growth. ( How? &gt;&gt;&gt;1 /0.1 = 10&nbsp;, 1/0.01 = 100&nbsp;, 1/0.001 =1000 are factor of 10 with growth of 10²)&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Now let’s see the performance of our model with different learning_rate.</strong>&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>Note: In our case performance of model means <strong>accuracy</strong> and <strong>speed</strong> of network</p></blockquote>
<!-- /wp:quote -->

<!-- wp:paragraph -->
<p><strong>Model_0 =&gt; learning_rate&nbsp;: 0.1</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/3a86a-1kdzz6rxfkzvcrau_olo4jw.png" alt=""/><figcaption>Training [lr =&nbsp;0.1]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/b35ac-1hl2_s4zfety9jbevrztfgq.png" alt=""/><figcaption>Testing [lr =&nbsp;0.1]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong>Model_1 =&gt; learning_rate&nbsp;:0.01</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/8e26a-1bdlx40eugpfvbwinqk0fnq.png" alt=""/><figcaption>Training [lr =&nbsp;0.01]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/ba2a8-1dyw7snhhpk7nxvxdwb8sxw.png" alt=""/><figcaption>Testing [lr&nbsp;=0.01]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong>Model_2 =&gt; learning_rate&nbsp;: 0.001</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/41a6e-1oecrmgrskx1srivnod_jig.png" alt=""/><figcaption>Training [lr =&nbsp;0.001]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/5f914-1lqt3nfmdljalo5dofeorpa.png" alt=""/><figcaption>Testing [lr&nbsp;=0.001]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Please observe the accuracy and speed of different models and you will get the idea of learning rate perfectly.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":4} -->
<h4><strong><em>What is Noise in terms of “text analysis”? and, why its NeuralNet approach is different than other ML&nbsp;methods?</em></strong></h4>
<!-- /wp:heading -->

<!-- wp:list -->
<ul><li>In, Normal Machine learning algorithms where we use NLTK, we have the leverage to remove some common words which haven’t any specific meaning (in term of our classification Objective). Means some words such as prepositions/ verbs/ spaces/ special characters.</li><li>But, for neural networks, we haven’t much leverage. Yes, we can use it during preprocessing. But, with a large amount of data is it really possible? (in reference to computational and time complexity).</li><li>Well, to resolve this complexity we have another way, I first display noise reduction hypothesis and then apply to the pre-trained model from Part-2</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Let’s first observe common words used to train our network…</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/665a7-1jvi0nfiwwjvvtka9rcgapa.png" alt=""/><figcaption>Most common words in review [as&nbsp;INPUT]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Our most counted words are without meaning.&nbsp;, but for Neural Network they give more importance when they multiply with weight(w) and give resultant output (y) for next neighboring neurons.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>And, as <strong>a result,</strong> make our model weaker during prediction as they <strong>remove focus(attention)</strong> from more meaningful (positive/negative) sentimental words.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>So, Now What is Solution: →</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>Hypothesis_1&nbsp;:</strong></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>What if we just use words included for a particular review out of our dictionary of words. And, do not think about “How many time such word/ words came in such review</p>
<!-- /wp:paragraph -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>What we will do?: <strong><em>Beside word count, we just measure word existance</em></strong></p></blockquote>
<!-- /wp:quote -->

<!-- wp:paragraph -->
<p>for this, I am changing our update_input_layer&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:syntaxhighlighter/code -->
<pre class="wp-block-syntaxhighlighter-code">def update_input_layer(self,reviews):                                       # count word uniqueness and return layer_0 as input_layer        self.layer_0 *= 0         for word in reviews.split(" "):          if(word in self.word2index.keys()):            self.layer_0[0][self.word2index[word]] +=1 </pre>
<!-- /wp:syntaxhighlighter/code -->

<!-- wp:syntaxhighlighter/code -->
<pre class="wp-block-syntaxhighlighter-code">=====================&lt; AFTER UPDATE:> ============================def update_input_layer(self,reviews):                                       # count word uniqueness and return layer_0 as input_layer        self.layer_0 *= 0         for word in reviews.split(" "):          if(word in self.word2index.keys()):            self.layer_0[0][self.word2index[word]] =1</pre>
<!-- /wp:syntaxhighlighter/code -->

<!-- wp:paragraph -->
<p>Here, besides counting words we are just using existence by 1 and if a word is not in the review then it becomes 0 so now input array should look like this&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:syntaxhighlighter/code -->
<pre class="wp-block-syntaxhighlighter-code">old_input = array[18,0,0,27,3,8,0,0,2,0]</pre>
<!-- /wp:syntaxhighlighter/code -->

<!-- wp:syntaxhighlighter/code -->
<pre class="wp-block-syntaxhighlighter-code">====================&lt; AFTER UPDATE > ==============================new_input = array[1,0,0,1,1,1,0,0,1,0]</pre>
<!-- /wp:syntaxhighlighter/code -->

<!-- wp:paragraph -->
<p>Let’s check Model performance improvement…</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/c6f14-1w1nldb8vmgthnkqoyc7tlw.png" alt=""/><figcaption>Noise Reduction hypothesis_1 Training [lr = 0.001] &gt;&gt;<strong> Improvement = 19 % Accuracy&nbsp;Increase</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/e3fd1-1ns6vdjvyipulbg91ap0d4a.png" alt=""/><figcaption>Noise Reduction hypothesis_1 Testing [lr = 0.001] &gt;&gt;<strong> Improvement = 7% Accuracy&nbsp;Increase&nbsp;</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>Great!, Until now, we changed our neural network performance <strong>in term of accuracy</strong> but as I mensioned before our <strong>speed of training and testing (reviews/sec) is not that much efficient</strong>.</p></blockquote>
<!-- /wp:quote -->

<!-- wp:paragraph -->
<p>To improve it we need to think on an operational level and here we need real <strong><em>Mathematic skills and aptitude…</em></strong></p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":4} -->
<h4><strong>Hypothesis 2:</strong></h4>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Our operation is started form layer_0 which actually has a long array of 0s and 1s, with a length of 74,074 words. Wow,<em> but in reality, </em><strong><em>most companies request customer’s reviews within 200 to 500 word</em>s.</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Even, if we actually see then there are fewer unique words than the actual total amount. In such a situation, the main focus is <strong>“Why we need to work with 74,074 words for observing one review with 200 -500 words?</strong> when the concern is about to compare the words with the positive or negative part of the dictionary, we can do it within the training of the model during prediction.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Solution:&nbsp;</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>During the creation of layer_1, input layer_0 multiply with the weight matrix, but these calculations are with 1 and 0 only! means (1*w = w, 0*w =0). So, it's better to update weight directly to layer_1 as per index which has 1s.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Why we do so?</strong>&nbsp;: Well, the reason is Computation power/ (complexity calculation)</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Normal Calculations&nbsp;:</strong> # operations = np.(input_layer(74074),layer_1[0])&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Now, calculations after applying hypothesis 2:</strong> layer_1 directly formed by&nbsp;… indices = [0,1,1,0,0,1]&nbsp;… where 1s exist…</p>
<!-- /wp:paragraph -->

<!-- wp:syntaxhighlighter/code -->
<pre class="wp-block-syntaxhighlighter-code">for index in indices:    layer_1 += (1 * weights_0_1[index])</pre>
<!-- /wp:syntaxhighlighter/code -->

<!-- wp:paragraph -->
<p><strong>So, # opreations = np.((200-500)&nbsp;,layer_1[0])</strong></p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/4c3ea-1ievzdqtaqoggqjmc4dmxvq.png" alt=""/><figcaption>Hypothesis Unit Testing [without&nbsp;hypo_2]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/3f349-10kd1-ed4bxneuvd5ze70sa.png" alt=""/><figcaption>Hypothesis Unit Testing [applied&nbsp;hypo_2]</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>See, Same output in each case. Note:</strong> here, layer_1 update also multiply by 1 so if we remove it, it will not change the output.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>Observation: We can save time and computation power. Speed of network will be improved.</p></blockquote>
<!-- /wp:quote -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/a0ff9-1rdvnibiyblzcsn6oornlkg.png" alt=""/><figcaption>Hypothesis_2&nbsp;: Training Speed <strong>Improvement by +950 (reviews/sec)</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/0aaf7-1elitdpvglsxh42cgv8d5bg.png" alt=""/><figcaption>Hypothesis_2&nbsp;: Testing Speed <strong>Improvement by +1500 (reviews/sec)</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p>In hypo_1 we just equalize counts of a word but if we can remove those words, it will give us further improvement in the network. so our next hypothesis_3 is an improved version of hypothesis_1. Let’s first try to understand.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>During data processing, we find pos_neg_ratio for getting positive and negative review words. as below, their ratios are quite far from 1, proved the most important words.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/d5145-1vstkysniwzmw5lp1xgzmvw.png" alt=""/><figcaption>Important words under Positive&nbsp;labels</figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/caac4-1ta8vozwftzcpgiixk4ysig.png" alt=""/><figcaption>Important words under Negative&nbsp;labels</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Let’s first visualize the distribution of such words&nbsp;…</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/63e5a-1xcp-jg7uhnexqwcaxcz1aq.png" alt=""/><figcaption>Words pos/neg. distribution</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Visualization Observation:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The X-axis shows probabilities, which means the positive and negative words are near to the end and their counts are quite low compares to common words.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Y-axis shows counts, and the highest counts are for the probability near to 0 or area between (-1,1). This graph resembles a normal distribution.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Repetition:</strong> Our common words have more counts and also exists with both negative and positive labels. So, their predictive power is less than unique words that exist in opposite labels.</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/59be2-1gmsco-8tvau-mwb9z5lpca.png" alt=""/><figcaption>Zitvian Distribution</figcaption></figure>
<!-- /wp:image -->

<!-- wp:list -->
<ul><li>Its <strong>Zitvian Distribution</strong> graph in which each line represents <strong>corpus (word collection)</strong>, represents that there are so many fewer words that dominate in text.&nbsp;</li><li>From, integration of both graphs we should conclude that “Those words around 1 in normal distribution dominate our text field. so, better to remove them by <strong><em>filtering pos_neg_ratio by the range and minimum counts value. This is the main idea behind our next hypothesis_3.&nbsp;</em></strong></li></ul>
<!-- /wp:list -->

<!-- wp:quote -->
<blockquote class="wp-block-quote"><p>We need to change prepocessing step to add new parameters called (polarity_cutoff, min_count)</p></blockquote>
<!-- /wp:quote -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/74908-1wgseh0m6mmbdqtj_tlgw8g.png" alt=""/><figcaption>Preprocessing_stage — update</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Please, refer to part_1 for preprocessing stage understanding. Now, we just do two main changes.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol><li>&nbsp;<strong>min_count </strong>&gt;&gt; we just count words, which are repeated more than a certain amount, will give us the advantage of neglecting some words such as industry name, type of product, company or brand name, distributor name, etc.</li><li><strong>polarity_cutoff &gt;&gt;</strong> This is one of the filters which decides the range of word from distribution graph. widen the range of negligence gives less input to the neural network. It improves speed but also has a chance of losing accuracy.&nbsp;</li></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>The following graph will give you a better idea…</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/9b3c7-1bpq91vacmuswv-c6lj07vq.jpeg" alt=""/><figcaption>Polarity_off in word distribution.</figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>In<strong><em> Range -1, less word is neglected will give less speed than Range-2 polarity_cutoff</em></strong>. but, speed may be affected easily.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Let’s check model performance…</p>
<!-- /wp:paragraph -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/bfd33-1ngzntfvia7lrlbxarb41lq.png" alt=""/><figcaption>Training Performance speed <strong>improved by +2000 (reviews/sec)</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:image -->
<figure class="wp-block-image"><img src="http://theprojectpages.files.wordpress.com/2020/05/d845f-1nkzywpmx0j45kro1p43wkw.png" alt=""/><figcaption>Testing Performance speed <strong>improved by +2100 (reviews/sec)</strong></figcaption></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><strong>Combined Evaluation:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here, <strong><em>“speed increase 3 times with the cost of 4% of accuracy”.</em></strong> Here, specifically, we can tolerate the accuracy of a model because when we discuss <strong>model performance with immense data </strong>(big data world) then the main task is to work with<strong> data optimization, and for that, if we get minimum tolerable accuracy </strong>which helps us to decide the objective. then, I think it's <strong><em>better to choose speed over accuracy</em></strong>. (Not always, think about high precision of data such as healthcare, aviation tech., public security etc…)</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For improving this scenario, companies can use some other features such as star rating buttons, emojis with text data. <strong>By integrating these data with text reviews, the “DS team” can improve their confidence level</strong>.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p>Thank you for reading. I try my best, still if you have any suggestions for me. Please, let me in a comment. And, if you like my work then please show your sentiment by giving me “clap”, it will keep me motivated.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The motto of my life:<strong><em> “Keep Learning, Enjoy Empowering”</em></strong></p>
<!-- /wp:paragraph -->

<!-- wp:separator {"color":"vivid-red","className":"is-style-wide"} -->
<hr class="wp-block-separator has-text-color has-background has-vivid-red-background-color has-vivid-red-color is-style-wide"/>
<!-- /wp:separator -->

<!-- wp:block {"ref":1522} /-->
