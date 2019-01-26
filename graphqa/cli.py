# -*- coding: utf-8 -*-

"""Console script for graphqa."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for graphqa."""
    click.echo("A simple code to translate English to Cypher "
               "graphqa.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
