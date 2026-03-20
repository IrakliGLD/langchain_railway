-- Supabase / PostgreSQL schema for vector-backed document retrieval.
-- Apply manually in Supabase SQL editor or through your deployment workflow.

create extension if not exists vector;
create extension if not exists pgcrypto;

create schema if not exists knowledge;

create table if not exists knowledge.documents (
    id uuid primary key default gen_random_uuid(),
    source_key text not null unique,
    title text not null,
    document_type text not null default '',
    issuer text not null default '',
    language text not null default 'en',
    source_url text null,
    storage_path text null,
    logical_key text not null default '',
    published_date date null,
    effective_date date null,
    effective_end_date date null,
    version_label text null,
    is_latest boolean not null default true,
    is_active boolean not null default true,
    abolished boolean not null default false,
    supersedes_document_id uuid null references knowledge.documents(id) on delete set null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table if exists knowledge.documents
    add column if not exists logical_key text not null default '';

alter table if exists knowledge.documents
    add column if not exists effective_end_date date null;

alter table if exists knowledge.documents
    add column if not exists is_latest boolean not null default true;

alter table if exists knowledge.documents
    add column if not exists abolished boolean not null default false;

alter table if exists knowledge.documents
    add column if not exists supersedes_document_id uuid null references knowledge.documents(id) on delete set null;

create table if not exists knowledge.document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references knowledge.documents(id) on delete cascade,
    chunk_index integer not null,
    section_title text not null default '',
    section_path text not null default '',
    page_start integer null,
    page_end integer null,
    text_content text not null,
    token_count integer not null default 0,
    language text not null default 'en',
    topics jsonb not null default '[]'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (document_id, chunk_index)
);

create index if not exists idx_knowledge_documents_active
    on knowledge.documents (is_active, document_type, language);

create index if not exists idx_knowledge_documents_lifecycle
    on knowledge.documents (logical_key, is_latest, abolished, effective_date, effective_end_date);

create index if not exists idx_knowledge_chunks_document
    on knowledge.document_chunks (document_id, chunk_index);

create index if not exists idx_knowledge_chunks_topics
    on knowledge.document_chunks using gin (topics);

create index if not exists idx_knowledge_chunks_metadata
    on knowledge.document_chunks using gin (metadata);

create index if not exists idx_knowledge_chunks_embedding
    on knowledge.document_chunks
    using hnsw (embedding vector_cosine_ops);
