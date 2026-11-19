# ADR 001: Uso de PostgreSQL com pgvector como Armazenamento Unificado

## Status
Aceito

## Contexto
O LicitAI precisa armazenar metadados relacionais estruturados de contratações públicas (CNPJ do órgão, datas, valores, status de processamento) e embeddings vetoriais dos trechos de editais em PDF, suportando busca híbrida (semântica + palavras-chave exatas para termos jurídicos e códigos de itens).

## Decisão
Adotar PostgreSQL 16 com a extensão `pgvector` e busca textual nativa (`tsvector` com configuração `pt_unaccent`), em vez de introduzir um banco vetorial dedicado (como Qdrant, Pinecone ou ChromaDB).

## Consequências
### Positivas
- **Consistência Transacional (ACID):** Metadados do edital e seus chunks vetoriais são persistidos ou revertidos na mesma transação.
- **Busca Híbrida Nativa:** Permite executar filtragem relacional (`WHERE uf = 'PE'`), lexical (`ts_rank`) e vetorial (`<=>`) em uma única consulta SQL.
- **Simplicidade Operacional:** Apenas um serviço de banco de dados para gerenciar no Docker Compose e em ambientes de nuvem.

### Negativas / Trade-offs
- Em escalas superiores a dezenas de milhões de vetores com alta concorrência de escrita, índices HNSW no PostgreSQL exigem ajuste fino de memória (`maintenance_work_mem`). Para o escopo de auditoria de editais (centenas de milhares de chunks), o desempenho é sub-100ms.
