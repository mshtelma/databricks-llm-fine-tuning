# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Prompt Engineering with LLMs on Databricks
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC <img src="https://imageio.forbes.com/specials-images/imageserve/645c8a0c17d25e16f92796dd/0x0.jpg?format=jpg&width=1200" />
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC ### Intro
# MAGIC
# MAGIC [Prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering) or prompting is the process of structuring sentences so that they can be interpreted and understood by a [Large Language Model (LLM)](https://www.databricks.com/resources/ebook/tap-full-potential-llm) in such a way that its output is in accord with the user's intentions. A prompt can be a description of a desired output, such as `as write a limerick about chickens` or `What is the capital of Lithuania?`.
# MAGIC
# MAGIC #### Parameters
# MAGIC
# MAGIC Before we start with Prompt Engineering, let's understand some common LLM Settings which can be used to tweak our generation capabilities.
# MAGIC
# MAGIC * **Temperature**: In short, the lower the `temperature`, the more *deterministic* the results. Conversely, higher temperature values might lead to more randomness - which can be desired for more creative outputs.
# MAGIC   * **Low Temperature**: you might want to use lower temperature values for fact-based QA in order to increase the chances of correct and concise responses.
# MAGIC   * **High Temperature**: For creative tasks, such as script and novel writing, higher temperature values might be useful.
# MAGIC
# MAGIC * **top_k**: K in this case indicates the number of samples that we will take amongst generated tokens, where tokens are sorted in descending order according to their probability. Same as with `temperature`, the lower the value of K, the more deterministic our output tends to be.
# MAGIC * **top_p**: Similarly to `temperature`, use lower values in case you are looking for more deterministic responses, and higher values otherwise. In this case, *p* is a cumulative probability value, so \\(0 < p <= 1\\)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### LLMs & Probability Distributions
# MAGIC
# MAGIC Going a bit more into detail - LLMs operate on a sequence of tokens, mostly sub-word units. The set of possible tokens is called the vocabulary of the LLM.
# MAGIC
# MAGIC The LLM takes in an input sequence of tokens and then tries to predict the next token. It does this by generating a **discrete probability distribution** over all possible tokens, using the **softmax** function as the last layer of the network. This is the raw output of the LLM.
# MAGIC
# MAGIC For example, if we had a vocabulary size of 5, the output might look like this:
# MAGIC
# MAGIC \\(t{_0} → {0.4} \\)
# MAGIC <br/>
# MAGIC \\(t{_1} → {0.2} \\)
# MAGIC <br/>
# MAGIC \\(t{_2} → {0.2} \\)
# MAGIC <br/>
# MAGIC \\(t{_3} → {0.15} \\)
# MAGIC <br/>
# MAGIC \\(t{_4} → {0.05} \\)
# MAGIC
# MAGIC ####Top-K Sampling
# MAGIC
# MAGIC For sampling tokens using **Top-K**, we select a value for K. The first Kth most likely tokens will be selected. Probabilities will be normalized - so that they amount to 1 when summed - and one of the tokens will be selected by sampling from a multinomial distribution.
# MAGIC
# MAGIC ####Top-p Sampling
# MAGIC
# MAGIC Also known as *nucleus* sampling, with **Top-p** instead of picking a number of tokens, we select the least amount of tokens which is enough to "cover" a certain amount of probability defined by the parameter *p*.
# MAGIC
# MAGIC Going back to the 5 tokens from our example above along with their probabilities, if we sampled using *top-p* with *p = 0.5* we would end up with tokens *t0* and *t1*, since we get a cumulative probability of 0.6 when we sum each of their probabilities - and this is already greater than our *p* value (0.5).
# MAGIC
# MAGIC Similar to the strategy with *Top-K*, we normalize the probability values and sample from a multinomial distribution.
# MAGIC
# MAGIC ####Temperature
# MAGIC
# MAGIC While with **Top-K** and **Top-p** we are basically sampling from our tokens according to their probabilities, with Temperature actually affects the softmax function itself.
# MAGIC
# MAGIC As a quick recap - the standard/unit softmax function is defined as below:
# MAGIC
# MAGIC \\(sigma(x_i) = \\frac{e^{x_{i}}}{\sum_{j=1}^K e^{x_{j}}} \ \ \ for\ i=1,2,\dots,K\\)
# MAGIC
# MAGIC By introducing a **temperature** parameter in this equation, we want to be able to control the **probabilities** or the **certainty** that is output by the softmax function. In order to achieve that, (scaled) softmax function becomes:
# MAGIC
# MAGIC \\(sigma(x_i) = \frac{e^{\frac{x_{i}}T}}{\sum_{j=1}^K e^{\frac{x_{j}}T}} \ \ \ for\ i=1,2,\dots,K \\)
# MAGIC
# MAGIC For our new, scaled version of softmax, we can observe the following behaviors depending on the value of *T*:
# MAGIC
# MAGIC * If \\(0<T<1\\), then the \\(x_{i} \\) input values get pushed further away from 0 and differences are amplified.
# MAGIC * If \\(T>1\\), then the \\(x_{i} \\) input values get pushed toward 0 and differences are reduced.
# MAGIC
# MAGIC We can have an idea on what happens to our token probabilities once we apply different temperature values by looking at the sample plots below:
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC <img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/temperature.png?raw=true" style="width: 50%"/>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Reference
# MAGIC
# MAGIC <hr />
# MAGIC
# MAGIC * [Prompt Engineering Guide](https://www.promptingguide.ai/)
# MAGIC * [Token Selection Strategies: Top-K, Top-p and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/)
