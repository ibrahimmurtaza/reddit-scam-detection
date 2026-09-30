# Decouple the corpus from the analysis pipeline

Reddit's Data API Terms prohibit using Reddit content as an input for model training without Reddit's explicit consent, and prohibit research use outside the Reddit For Researchers program. Rather than block the project on an application we do not control, ingestion sits behind a `CorpusProvider` interface: the portfolio build supplies a synthetic corpus, and a real provider can be added later if authorisation is granted.

## Considered Options

- **Couple the pipeline directly to the Reddit API.** Rejected: no authorisation to train on the content, and the application outcome would gate all engineering.
- **Ingest from Arctic Shift**, a free unauthenticated archive. Rejected on the same grounds — free to download is not authorised to use, and routing through a third party obscures the question rather than answering it.
- **Change platforms.** Not rejected, but it trades away the domain instead of solving the access problem. The provider interface keeps it open as a later option.

## Consequences

The interface is a seam with only one implementation, which will look like over-engineering until a second one exists. That is the intent: it is what makes the synthetic and real paths swappable, and it keeps the authorisation question on a parallel track rather than blocking the build.
