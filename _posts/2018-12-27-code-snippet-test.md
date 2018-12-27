---
layout: post
title: Code Snippet Test
tags: [ miscellaneous,  ]
---

Here is a code snippet:

```Python
df = pd.DataFrame(iris.data, columns=[iris.feature_names])
df['labels'] = iris.target
df.head()

df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'labels']
```
