# TACTIQ AI — Data Dictionary

## 1. Overview

This document describes the general structure and interpretation of the tracking data processed by TACTIQ AI.

It serves as technical documentation for the internal data representation, feature extraction process, and possession-estimation pipeline.

Competition-specific metadata, private identifiers, raw tracking records, and match-derived statistics are intentionally excluded.

## 2. Match and Team Representation

The system supports two opposing teams:

- **Home:** The designated home team.
- **Away:** The designated away team.

Team assignments are represented internally using normalized team identifiers.

The system should use the team mappings provided by the authorized input data rather than hard-coding match-specific team names or identifiers.

## 3. Tracking Objects

The tracking input may contain observations for:

- Players.
- The ball.
- Frame-level tracking metadata.

Each player observation represents a tracked object within a particular frame.

Tracking identifiers represent source-level object or tracklet references. They should not automatically be interpreted as permanent player identities unless the source explicitly guarantees identity continuity.

The ball observation may be available or missing in individual frames.

## 4. ATD Data Representation

The tracking-data loader processes a frame-oriented structure containing player observations and ball information.

Conceptually, a frame can contain:

- A frame identifier.
- Zero or more player observations.
- An optional ball observation.

Each tracked object may provide attributes such as:

- X coordinate.
- Y coordinate.
- Movement speed.
- Source-level tracking identifier.

Missing observations may be represented using non-numeric values such as `NaN`, depending on the source format.

The data loader is responsible for converting these values into a consistent internal representation.

## 5. Frame Indexing

The source tracking data and submission format may use different frame-index conventions.

The processing pipeline must apply the indexing convention specified by the official input and submission formats.

Frame alignment should be validated locally before generating the final submission.

Do not assume that source frame identifiers and submission indexes are interchangeable.

## 6. Coordinate System

Player and ball positions are represented using normalized pitch-relative coordinates.

The coordinate system is interpreted using the pitch dimensions provided by the authorized input metadata.

Coordinates should not automatically be clipped to a predefined range. Values outside the expected normalized boundaries may indicate source-specific coordinate behavior or tracking anomalies and should be handled deliberately.

### Distance Calculation

For normalized coordinate differences:

- Δx: Difference in horizontal normalized coordinates.
- Δy: Difference in vertical normalized coordinates.
- L: Pitch length in metres.
- W: Pitch width in metres.

The corresponding Euclidean distance is:

\[
d = \sqrt{(L \cdot \Delta x)^2 + (W \cdot \Delta y)^2}
\]

The pitch dimensions must come from authorized source metadata rather than hard-coded match-specific values.

## 7. Movement Speed

Tracking observations may include movement-speed metadata.

The processing pipeline should preserve the source unit and apply any required unit conversion explicitly.

Speed values should not be interpreted or recalculated without considering their source definitions and reliability.

Missing or anomalous speed values must be handled safely.

## 8. Homography

Homography information may be available to transform pitch-relative coordinates into image or broadcast-video pixel coordinates.

Its applicability depends on the coordinate systems and camera calibration associated with the input data.

Homography is not required for direct Euclidean distance calculations when player and ball coordinates are already expressed in the same pitch-relative coordinate system.

## 9. Smart Tagging

Smart Tagging provides automated contextual information about football events and match phases.

Such metadata may help describe the context surrounding a tracking observation.

Important considerations:

- Tags may overlap.
- Multiple events may be associated with the same period.
- Team information may be incomplete or conflicting.
- Automated tagging is not definitive possession ground truth.

Smart Tagging should therefore be treated as contextual evidence rather than an authoritative possession-label source.

## 10. Timeline and Synchronization

Different data sources may use different timelines, sampling conventions, or synchronization procedures.

The processing pipeline must identify the appropriate authorized source for the challenge and validate any required alignment.

Data from separate timelines must not be assumed to be synchronized automatically.

## 11. Data Quality and Missing Observations

The system must account for imperfect tracking information.

Potential issues include:

- Missing player observations.
- Missing ball observations.
- Inconsistent tracking identifiers.
- Invalid or non-finite coordinate values.
- Incomplete movement metadata.
- Coordinate anomalies.

Missing observations alone should not be treated as proof of a possession change or dead-ball event. A live loose or contested ball is not automatically `D`; `D` is reserved for ball-not-in-play state from the separate dead-ball layer.

## 12. Data Safety

The public repository is intended to contain source code, generic technical documentation, tests, and reproducible configuration files.

The following materials must remain outside the public repository unless explicitly authorized by the competition organizer:

- Raw competition tracking data.
- Match videos and broadcast recordings.
- Private annotations and ground-truth files.
- Match-specific identifiers and restricted metadata.
- Match-derived statistics, extracts, and outputs that could reconstruct restricted information.

All data processing must follow the official competition rules and permitted-use requirements.