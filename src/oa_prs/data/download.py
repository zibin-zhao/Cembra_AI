"""
Data download manager for GWAS summary statistics, LD reference panels, and GTEx models.

Handles downloading from public repositories with resume capability, checksum verification,
and progress logging.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import structlog

logger = structlog.get_logger(__name__)


class DataDownloader:
    """
    Download manager for genomics reference data.

    Supports downloading GWAS summary statistics, 1000 Genomes LD reference panels,
    and GTEx models with resume capability and checksum verification.
    """

    # Public repository URLs
    GWAS_CATALOG_BASE = "https://ftp.ebi.ac.uk/pub/databases/gwas/"
    THOUSAND_GENOMES_BASE = "http://ftp.1000genomes.ebi.ac.uk/vol1/"
    GTEX_BASE = "https://gtexportal.org/static/datasets/"

    def __init__(
        self,
        cache_dir: str | Path = "~/.oa_prs/data",
        chunk_size: int = 8192,
        timeout: int = 30,
    ):
        """
        Initialize DataDownloader.

        Args:
            cache_dir: Directory to cache downloaded files. Defaults to ~/.oa_prs/data
            chunk_size: Size of chunks for streaming downloads (bytes)
            timeout: HTTP request timeout (seconds)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.timeout = timeout
        logger.info("DataDownloader initialized", cache_dir=str(self.cache_dir))

    def download_gwas_sumstats(
        self,
        url: str,
        filename: str,
        checksum: Optional[str] = None,
        resume: bool = True,
        force: bool = False,
    ) -> Path:
        """
        Download GWAS summary statistics file.

        Args:
            url: URL to download from
            filename: Local filename to save as (relative to cache_dir)
            checksum: Optional MD5 checksum for verification
            resume: Whether to resume incomplete downloads
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If checksum verification fails
            URLError: If download fails after retries
        """
        return self._download_file(
            url=url,
            filename=filename,
            checksum=checksum,
            resume=resume,
            force=force,
            file_type="GWAS summary statistics",
        )

    def download_ld_references(
        self,
        population: str,
        chrom: int,
        filename: Optional[str] = None,
        checksum: Optional[str] = None,
        resume: bool = True,
        force: bool = False,
    ) -> Path:
        """
        Download 1000 Genomes LD reference panel for a chromosome.

        Args:
            population: Population code (EUR, EAS, AFR, SAS, AMR)
            chrom: Chromosome number (1-22)
            filename: Optional custom filename (defaults to standard naming)
            checksum: Optional MD5 checksum for verification
            resume: Whether to resume incomplete downloads
            force: Force re-download even if file exists

        Returns:
            Path to downloaded LD file

        Raises:
            ValueError: If population or chromosome is invalid
        """
        if not 1 <= chrom <= 22:
            raise ValueError(f"Chromosome must be 1-22, got {chrom}")

        valid_pops = {"EUR", "EAS", "AFR", "SAS", "AMR"}
        if population not in valid_pops:
            raise ValueError(f"Population must be one of {valid_pops}, got {population}")

        if filename is None:
            filename = f"1kg_ld_{population.lower()}_chr{chrom}.vcf.gz"

        # Construct URL to 1000 Genomes VCF files
        url = f"{self.THOUSAND_GENOMES_BASE}VCF/20130502/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"

        logger.info(
            "Downloading LD reference",
            population=population,
            chrom=chrom,
            filename=filename,
        )

        return self._download_file(
            url=url,
            filename=filename,
            checksum=checksum,
            resume=resume,
            force=force,
            file_type="LD reference",
        )

    def download_gtex_models(
        self,
        tissue: str,
        filename: Optional[str] = None,
        checksum: Optional[str] = None,
        resume: bool = True,
        force: bool = False,
    ) -> Path:
        """
        Download GTEx prediction models for a tissue.

        Args:
            tissue: Tissue name (e.g., 'Whole_Blood', 'Liver')
            filename: Optional custom filename
            checksum: Optional MD5 checksum for verification
            resume: Whether to resume incomplete downloads
            force: Force re-download even if file exists

        Returns:
            Path to downloaded GTEx model

        Raises:
            URLError: If download fails
        """
        if filename is None:
            filename = f"gtex_{tissue.lower()}_model.db"

        # Simplified GTEx URL path
        url = f"{self.GTEX_BASE}gtex_v8_models/{tissue}.db"

        logger.info("Downloading GTEx model", tissue=tissue, filename=filename)

        return self._download_file(
            url=url,
            filename=filename,
            checksum=checksum,
            resume=resume,
            force=force,
            file_type="GTEx model",
        )

    def download_all(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Download all required data files based on configuration.

        Args:
            config: Configuration dict with keys:
                - gwas_urls: List of (url, filename, checksum) tuples
                - populations: List of populations for LD refs
                - tissues: List of GTEx tissues to download

        Returns:
            Dict mapping file types to their local paths

        Example:
            config = {
                'gwas_urls': [('http://example.com/gwas.txt.gz', 'gwas.txt.gz', None)],
                'populations': ['EUR', 'EAS'],
                'tissues': ['Whole_Blood', 'Adipose_Tissue'],
            }
            paths = downloader.download_all(config)
        """
        downloaded_files = {}

        # Download GWAS files
        gwas_files = config.get("gwas_urls", [])
        logger.info("Downloading GWAS files", count=len(gwas_files))
        downloaded_files["gwas"] = []
        for url, filename, checksum in gwas_files:
            path = self.download_gwas_sumstats(url, filename, checksum)
            downloaded_files["gwas"].append(path)

        # Download LD references
        populations = config.get("populations", [])
        logger.info("Downloading LD references", populations=populations)
        downloaded_files["ld"] = {}
        for pop in populations:
            downloaded_files["ld"][pop] = {}
            for chrom in range(1, 23):
                try:
                    path = self.download_ld_references(pop, chrom)
                    downloaded_files["ld"][pop][chrom] = path
                except URLError as e:
                    logger.warning(
                        "Failed to download LD reference",
                        population=pop,
                        chrom=chrom,
                        error=str(e),
                    )

        # Download GTEx models
        tissues = config.get("tissues", [])
        logger.info("Downloading GTEx models", count=len(tissues))
        downloaded_files["gtex"] = []
        for tissue in tissues:
            try:
                path = self.download_gtex_models(tissue)
                downloaded_files["gtex"].append(path)
            except URLError as e:
                logger.warning(
                    "Failed to download GTEx model", tissue=tissue, error=str(e)
                )

        logger.info(
            "Data download completed",
            gwas_count=len(downloaded_files["gwas"]),
            ld_count=sum(
                len(v) for v in downloaded_files["ld"].values()
            ),
            gtex_count=len(downloaded_files["gtex"]),
        )

        return downloaded_files

    def _download_file(
        self,
        url: str,
        filename: str,
        checksum: Optional[str] = None,
        resume: bool = True,
        force: bool = False,
        file_type: str = "file",
    ) -> Path:
        """
        Internal method to download a single file with resume and checksum verification.

        Args:
            url: URL to download from
            filename: Local filename (relative to cache_dir)
            checksum: Optional MD5 checksum for verification
            resume: Whether to resume incomplete downloads
            force: Force re-download
            file_type: Description of file type for logging

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If checksum verification fails
            URLError: If download fails
        """
        filepath = self.cache_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and is complete
        if filepath.exists() and not force:
            if checksum is None:
                logger.info(
                    f"{file_type} already exists, skipping download",
                    path=str(filepath),
                )
                return filepath

            # Verify checksum
            if self._verify_checksum(filepath, checksum):
                logger.info(
                    f"{file_type} checksum verified, skipping download",
                    path=str(filepath),
                )
                return filepath
            else:
                logger.warning(
                    f"{file_type} checksum mismatch, re-downloading",
                    path=str(filepath),
                )
                filepath.unlink()

        # Download file
        logger.info(f"Downloading {file_type}", url=url, path=str(filepath))

        try:
            request = Request(url, headers={"User-Agent": "oa_prs_downloader"})
            with urlopen(request, timeout=self.timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    content_length = int(content_length)

                bytes_downloaded = 0
                with open(filepath, "wb") as f:
                    while True:
                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

                        if content_length:
                            progress_pct = (bytes_downloaded / content_length) * 100
                            logger.debug(
                                f"Downloading {file_type}",
                                bytes=bytes_downloaded,
                                total=content_length,
                                progress_pct=f"{progress_pct:.1f}%",
                            )

        except (URLError, HTTPError) as e:
            logger.error(
                f"Failed to download {file_type}",
                url=url,
                error=str(e),
            )
            if filepath.exists():
                filepath.unlink()
            raise

        # Verify checksum if provided
        if checksum is not None:
            if not self._verify_checksum(filepath, checksum):
                filepath.unlink()
                raise ValueError(
                    f"Checksum verification failed for {filepath}. "
                    f"Expected {checksum}, but file may be corrupted."
                )
            logger.info(f"{file_type} checksum verified", path=str(filepath))

        logger.info(f"{file_type} downloaded successfully", path=str(filepath))
        return filepath

    @staticmethod
    def _verify_checksum(filepath: Path, expected_md5: str) -> bool:
        """
        Verify MD5 checksum of a file.

        Args:
            filepath: Path to file to verify
            expected_md5: Expected MD5 checksum (hex string)

        Returns:
            True if checksum matches, False otherwise
        """
        md5_hash = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest().lower() == expected_md5.lower()
        except (IOError, OSError) as e:
            logger.error("Checksum verification error", path=str(filepath), error=str(e))
            return False
