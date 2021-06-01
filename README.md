 Lead Calculation with Rule-Based Classification

![0_PNP8NN84PaZP7VSs](https://user-images.githubusercontent.com/73841520/120351719-c5c35680-c308-11eb-9906-ac4a3a4c3b64.jpg)


What is the problem?
- A game company wants to create new customer definitions (personas) based on level by using some features of its customers, to create segments according to these new customer definitions and to estimate how much new customers can earn according to these segments.

Dataset Story
- The Persona.csv dataset contains the prices of the products sold by an international game company and some demographic information of the users who buy these products.
The dataset consists of records created in each sales transaction.
This means the table is not deduplicated.
In other words, a user with certain demographics may have made more than one purchase.

VARIABLES
- PRICE: CUSTOMER EXPENDITURE
- SOURCE: CUSTOMER CONNECTED DEVICE TYPE
- GENDER: CUSTOMER'S GENDER
- COUNTRY: CUSTOMER COUNTRY
- AGE: CUSTOMER AGE
