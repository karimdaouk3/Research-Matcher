Goal: Create a website that allows users to input a research project idea, and outputs relevant faculty that the user could conduct their research with.

Web Scraping

Web scraped from the SEAS directory page. The website is not static, because new staff members are loaded onto the page each time the ‘next page’ button is clicked. 
To get all of the faculty research summaries, the link to each professor’s biography page is first scraped using a web controller (Selenium). Upon loading the SEAS directory page, the ‘faculty’ filter is selected. Then the name of each faculty member along with each of their corresponding links are scraped and put into a pandas dataframe. The ‘next page’ button is clicked, and the scraping process is repeated until the next page button is no longer on the page. 
At this point, all of the faculty names and the links to their biography pages are stored in a dataframe.
Next, the dataframe is iterated through. For each faculty member, their link is visited, and the research summary portion of the page is scraped. The text is stored as a field for each faculty entry in the dataframe. 

Text Vectorizing

In order to compare the faculty research summaries with user input, the text must be turned into a vector.
This is done using a pretrained large language model, all-MiniLM-L12-v2.
This model converts text into a 384 dimensional unit vector, which encodes the meaning on the text.
The model does not encode semantic information, meaning that its encoding of a word is not dependent on its context.
The model is downloaded locally, and does not require API access
For each faculty member, their research summary is inputted in the model, and its vector output is stored as a field for each faculty entry in the dataframe
The resulting dataframe is stored as a json, to be loaded upon starting of the app
When a user inputs a research project idea, the input text is also converted to a vector

Text Similarity Comparison

Two texts with similar words yield similar encoded vectors
Thus, text similarity is evaluated using the similarity of the encoded vectors
Since all encoded vectors are one unit long, their similarity can be evaluated by finding how close they are in direction
To measure this, the dot product is calculated. Vectors with a dot product close to 1 are similar in meaning. Dot product values closer to 0 indicate less similarity.
The dot product of the encoded input vector is calculated with the encoded research summary vector for each faculty member. This dot product is stored as a new field in the dataframe. The dataframe is sorted in descending order by the dot product value.
As a result, the dataframe of faculty is sorted according to how similar their research summary is to a given input.
Flask App
Upon startup, the app downloads the content of the faculty json and stores it into a dataframe. The text vectorizer model is also initialized
Upon receiving a POST request (when the user submits an input), the input is vectorized. The dot product between the vectorized input and all encoded research summaries is calculated, and the faculty dataframe is sorted by relevance.
The website displays the top x most relevant faculty. The value of x is specified by the user.

Generate Title feature

The website provides an option to generate a title for the user’s research idea.
When the input is submitted, the app makes a request to the OpenAI API with the input and the following specification: "You are an assistant that generates a simple title for a research paper given a research proposal. Do not include the word title in the response."


