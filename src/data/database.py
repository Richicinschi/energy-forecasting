"""
database.py — SQLAlchemy models and database utilities for the energy forecasting project.

Tables:
    region_data      — D, DF, NG, TI per BA per hour (primary modelling table)
    fuel_type_data   — Hourly generation by fuel type per BA
    interchange_data — Hourly net flow between BA pairs
    sub_ba_data      — Hourly demand by subregion within each BA
    ingest_log       — Audit log of every ingestion run

Usage:
    from src.data.database import get_engine, create_all_tables, get_session
    engine = get_engine()
    create_all_tables(engine)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import (
    Column, DateTime, Float, Index, Integer, String, Text,
    UniqueConstraint, create_engine, event, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

class RegionData(Base):
    """
    Core modelling table: hourly D, DF, NG, TI per Balancing Authority.

    Columns matching EIA-930 region-data endpoint:
        period       — UTC hour (e.g. 2022-01-01 00:00:00+00:00)
        respondent   — BA code (MISO, PJM, ERCO, ...)
        type         — D | DF | NG | TI
        value_mwh    — Value in megawatt-hours
        is_anomaly   — Flagged by >N std from rolling 168h mean
        is_imputed   — Gap-filled by linear interpolation
    """
    __tablename__ = "region_data"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    period          = Column(DateTime(timezone=True), nullable=False)
    respondent      = Column(String(16), nullable=False)
    respondent_name = Column(String(128))
    type            = Column(String(4), nullable=False)
    type_name       = Column(String(64))
    value_mwh       = Column(Float)
    is_anomaly      = Column(Integer, default=0)   # 0/1 flag
    is_imputed      = Column(Integer, default=0)   # 0/1 flag

    __table_args__ = (
        UniqueConstraint("period", "respondent", "type", name="uq_region_period_ba_type"),
        Index("ix_region_respondent_type_period", "respondent", "type", "period"),
        Index("ix_region_period", "period"),
    )

    def __repr__(self) -> str:
        return f"<RegionData {self.respondent} {self.type} {self.period} {self.value_mwh}>"


class FuelTypeData(Base):
    """Hourly net generation by fuel type per BA (fuel-type-data endpoint)."""
    __tablename__ = "fuel_type_data"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    period          = Column(DateTime(timezone=True), nullable=False)
    respondent      = Column(String(16), nullable=False)
    respondent_name = Column(String(128))
    fueltype        = Column(String(8), nullable=False)
    type_name       = Column(String(64))
    value_mwh       = Column(Float)

    __table_args__ = (
        UniqueConstraint("period", "respondent", "fueltype", name="uq_fuel_period_ba_fuel"),
        Index("ix_fuel_respondent_fueltype_period", "respondent", "fueltype", "period"),
    )


class InterchangeData(Base):
    """Hourly net power flow between neighbouring BA pairs (interchange-data endpoint)."""
    __tablename__ = "interchange_data"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    period    = Column(DateTime(timezone=True), nullable=False)
    fromba    = Column(String(16), nullable=False)
    fromba_name = Column(String(128))
    toba      = Column(String(16), nullable=False)
    toba_name = Column(String(256))
    value_mwh = Column(Float)

    __table_args__ = (
        UniqueConstraint("period", "fromba", "toba", name="uq_interchange_period_pair"),
        Index("ix_interchange_fromba_period", "fromba", "period"),
    )


class SubBaData(Base):
    """Hourly demand by subregion within each BA (region-sub-ba-data endpoint)."""
    __tablename__ = "sub_ba_data"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    period    = Column(DateTime(timezone=True), nullable=False)
    subba     = Column(String(32), nullable=False)
    subba_name = Column(String(128))
    parent    = Column(String(16), nullable=False)
    value_mwh = Column(Float)

    __table_args__ = (
        UniqueConstraint("period", "subba", name="uq_subba_period_subba"),
        Index("ix_subba_parent_period", "parent", "period"),
    )


class IngestLog(Base):
    """Audit log of every ingestion run."""
    __tablename__ = "ingest_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_at       = Column(DateTime(timezone=True), nullable=False)
    endpoint     = Column(String(32))
    respondent   = Column(String(16))
    source_file  = Column(Text)
    rows_raw     = Column(Integer)
    rows_ingested= Column(Integer)
    rows_imputed = Column(Integer)
    rows_anomaly = Column(Integer)
    status       = Column(String(16))   # success | error
    error_msg    = Column(Text)


# ─────────────────────────────────────────────────────────────────────────────
# Engine / session factory
# ─────────────────────────────────────────────────────────────────────────────

def get_engine(db_url: str | None = None, echo: bool = False):
    """Create and return a SQLAlchemy engine.

    Defaults to SQLite at data/energy_forecasting.db.
    Enables WAL mode and foreign keys for SQLite.
    """
    if db_url is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.environ.get(
            "DATABASE_URL", "sqlite:///data/energy_forecasting.db"
        )

    # Ensure parent directory exists for SQLite
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(db_url, echo=echo, future=True)

    # SQLite performance tuning
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(conn, _record):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")   # 64 MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA foreign_keys=ON")

    return engine


def create_all_tables(engine) -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(engine)
    logger.info("All tables created (or already exist)")


def get_session(engine) -> Generator[Session, None, None]:
    """Context-manager-style session factory."""
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_table_counts(engine) -> dict[str, int]:
    """Return row counts for all main tables."""
    tables = ["region_data", "fuel_type_data", "interchange_data", "sub_ba_data"]
    counts = {}
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except Exception:
                counts[table] = -1
    return counts
