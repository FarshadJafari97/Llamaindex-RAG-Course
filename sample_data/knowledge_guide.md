# Northstar Knowledge Guide

## Retrieval basics

Retrieval augmented generation combines relevant document context with a language model.
A typical pipeline loads documents, splits them into nodes, creates embeddings, stores the nodes in an index, and retrieves similar nodes for a query.

## Recommended settings

Use a sentence splitter with a small chunk overlap when experimenting.
Set the number of retrieved documents with `similarity_top_k`.
